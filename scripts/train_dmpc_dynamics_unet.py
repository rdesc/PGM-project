import os
from dataclasses import dataclass, asdict
from diffuser.datasets import SequenceDataset
from diffusers import  UNet1DModel, DDPMScheduler
from tqdm import tqdm
from accelerate import Accelerator
from huggingface_hub import HfFolder, Repository, whoami
from tqdm.auto import tqdm
from pathlib import Path
import torch
import torch.nn.functional as F
from diffusers import DDPMPipeline
import accelerate
from diffuser.utils.rendering import MuJoCoRenderer
from diffusers.optimization import get_cosine_schedule_with_warmup, get_constant_schedule
from diffuser.utils import set_seed
import time
import yaml
import tyro
import wandb
import copy


def cycle(dl):
    while True:
        for data in dl:
            yield data

device = "cuda" if torch.cuda.is_available() else "cpu"


def generate_samples(config, conditioning, model, dataset, scheduler, use_pipeline=False, obs_only=True,):
    generator = torch.Generator(device=device)
    shape = (config.eval_batch_size, config.horizon, dataset.observation_dim + dataset.action_dim,)
    x = torch.randn(shape, device=device, generator=generator).to(device)
    s_t, h_t, actions = conditioning
    if use_pipeline:
        for i, t in enumerate(scheduler.timesteps):
            timesteps = torch.full((config.eval_batch_size,), t, device=device, dtype=torch.long)
            model_input = scheduler.scale_model_input(x, t)
            
            model_input[:, config.history, dataset.action_dim:] = s_t
            model_input[:, :config.history, :] = h_t.permute(0, 2, 1)
            model_input[:, :, :dataset.action_dim] = actions[:, :, :-1].permute(0, 2, 1)

            with torch.no_grad():
                noise_pred = model(model_input.permute(0, 2, 1), timesteps).sample
                noise_pred = noise_pred.permute(0, 2, 1) # needed to match model params to original
            x = scheduler.step(noise_pred, t, x).prev_sample
        
        if config.use_conditioning_for_sampling:
            x[:, config.history, dataset.action_dim:] = s_t
            x[:, :config.history, :] = h_t.permute(0, 2, 1)
            x[:, :, :dataset.action_dim] = actions[:, :, :-1].permute(0, 2, 1)
        
    else:
        # sample random initial noise vector
        eta = 1.0 # noise factor for sampling reconstructed state

        # run the diffusion process
        x[:, config.history, dataset.action_dim:] = s_t
        x[:, :config.history, :] = h_t.permute(0, 2, 1)
        x[:, :, :dataset.action_dim] = actions[:, :, :-1].permute(0, 2, 1)

        for i in tqdm(scheduler.timesteps):

            # create batch of timesteps to pass into model
            timesteps = torch.full((config.eval_batch_size,), i, device=device, dtype=torch.long)

            # 1. generate prediction from model
            with torch.no_grad():
                residual = model(x.permute(0, 2, 1), timesteps).sample
                residual = residual.permute(0, 2, 1) # needed to match model params to original

            # 2. use the model prediction to reconstruct an observation (de-noise)
            obs_reconstruct = scheduler.step(residual, i, x)["prev_sample"]

            # 3. [optional] add posterior noise to the sample
            if eta > 0:
                noise = torch.randn(obs_reconstruct.shape, generator=generator, device=device)
                posterior_variance = scheduler._get_variance(i) # * noise
                # no noise when t == 0
                # NOTE: original implementation missing sqrt on posterior_variance
                obs_reconstruct = obs_reconstruct + int(i>0) * (0.5 * posterior_variance) * eta* noise  # MJ had as log var, exponentiated
            
            if config.use_conditioning_for_sampling:
                obs_reconstruct[:, config.history, dataset.action_dim:] = s_t
                obs_reconstruct[:, :config.history, :] = h_t.permute(0, 2, 1)
                obs_reconstruct[:, :, :dataset.action_dim] = actions[:, :, :-1].permute(0, 2, 1)

            # 4. apply conditions to the trajectory
            # obs_reconstruct_postcond = reset_x0(obs_reconstruct, conditions, action_dim)
            x = obs_reconstruct
    if obs_only:
        x = x[:, :, dataset.action_dim:]
    return x


def save_configs(configs, save_path):
    file_path = f"{save_path}/config.yaml"
    with open(file_path, 'w') as f:
        f.write(yaml.dump(asdict(configs)))


@dataclass
class TrainingConfig:
    env_id: str = "hopper-medium-v2"
    train_batch_size: int = 16
    eval_batch_size: int = 1  # how many images to sample during evaluation
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    lr_warmup_steps: int = 500
    cosine_warmup: bool = True
    num_train_timesteps: int = 100
    n_train_steps: int = int(200e3)
    n_train_step_per_epoch: int = 10_000
    mixed_precision: str = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir: str = "ddpm-butterflies-128"  # the model name locally and on the HF Hub
    horizon: int = 32
    push_to_hub: bool = True  # whether to upload the saved model to the HF Hub
    hub_private_repo: bool = False
    overwrite_output_dir: bool = True  # overwrite the old model when re-running the notebook
    seed: int = 0
    use_sample_hf_for_render: bool= False
    use_conditioning_for_sampling: bool = True
    checkpointing_freq: int = 20_000
    wandb_track: bool = True
    num_workers: int = 1
    torch_compile: bool = True
    model_type: str = 'dynamics_unet'
    use_original_config: bool = True
    train_on_one_traj: bool = False
    use_grad_clip: bool = False # try training without gradient 
    weight_decay: float = 0.01 # try training with weight decay zero
    ema_decay: float = 0.995
    save_ema: bool = True
    update_ema_every: int = 10
    history: int = 1  # hardcode to 1 for now
    render_freq: int = 4200


def update_ema(ema_model, model, decay):
    with torch.no_grad():  
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(decay).add_(param.data, alpha=(1 - decay))
    for ema_buffer, buffer in zip(ema_model.buffers(), model.buffers()):
        ema_buffer.copy_(buffer)


def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler, renderer, save_path):
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs"),
    )
    if accelerator.is_main_process:
        # if config.push_to_hub:
            # repo_name = get_full_repo_name(Path(config.output_dir).name)
            # repo = Repository(config.output_dir, clone_from=repo_name)
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        accelerator.init_trackers("train_example")

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    # create ema model
    model_ema = copy.deepcopy(model)
    for param in model_ema.parameters():
        param.requires_grad_(False)

    checkpoint_path = f"{save_path}/checkpoints"
    os.makedirs(checkpoint_path)
    save_configs(config, save_path)
    # save scheduler as well for evaluation
    scheduler.save_pretrained(save_path)
    
    one_trajectory = None
    # Now you train the model
    for i in tqdm(range(int(config.n_train_steps))):
        # progress_bar = tqdm(total=int(config.n_train_steps), disable=not accelerator.is_local_main_process)
        # progress_bar.set_description(f"Epoch {i}")
        batch = next(train_dataloader)
        trajectories = batch.trajectories.to(device)
        if config.train_on_one_traj:
            if one_trajectory is None:
                one_trajectory = torch.clone(trajectories[0][None, :])
            trajectories = one_trajectory

        trajectories = torch.permute(trajectories, (0,2,1) )  # shape (bs, state_dim + action_dim, horizon + history)
        actions = trajectories[:, :dataset.action_dim, :]  # shape (bs, action_dim, horizon)

        # extract the current state and history
        s_t = trajectories[:, dataset.action_dim:, config.history]  # shape (bs, state_dim)
        h_t = trajectories[:, :, :config.history]
        
        # Sample noise to add to the images
        bs = trajectories.shape[0]
        noise = torch.randn((trajectories.shape)).to(trajectories.device)  # this is the wrong dim right now

        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps, (bs,), device=trajectories.device
        ).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectories = noise_scheduler.add_noise(trajectories, noise, timesteps)

        # add conditioning for the first state s_t, history h_t, and actions
        noisy_trajectories[:, dataset.action_dim:, config.history] = s_t
        noisy_trajectories[:, :, :config.history] = h_t
        noisy_trajectories[:, :dataset.action_dim, :] = actions

        with accelerator.accumulate(model):
            # Predict the noise residual
            if config.use_original_config:
                sample_pred = model(noisy_trajectories[:, :, :-1], timesteps, return_dict=False)[0]  # clip the last timestep

            # else:
            #     noise_pred = model(noisy_trajectories, timesteps, return_dict=False)[0]

            # apply conditioning
            sample_pred[:, dataset.action_dim:, config.history] = s_t
            sample_pred[:, :, :config.history] = h_t
            sample_pred[:, :dataset.action_dim, :] = actions[:, :, :-1]
            
            if config.use_original_config:
                loss = F.mse_loss(sample_pred, trajectories[:, :, :-1], reduction='none')
            # else:
            #     # loss = F.mse_loss(noise_pred, noise)
            #     loss = F.mse_loss(noise_pred, trajectories, reduction='none')

            # compute loss just for immediate state prediction, for logging
            s0_loss = loss[:, :, 0].mean().detach().item()
            weighted_loss = loss.mean()
            accelerator.backward(weighted_loss)

            if config.use_grad_clip:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # update EMA
            if (i + 1) % config.update_ema_every == 0:
                update_ema(model_ema, model, config.ema_decay)

            # progress_bar.update(1)
            logs = {"loss": weighted_loss.detach().item(), "s0_loss": s0_loss, "lr": lr_scheduler.get_last_lr()[0], "step": i}
            if config.wandb_track:
                wandb.log(logs, step=i)
            # progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=i)
            # global_step += 1

        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if (i+1) % config.checkpointing_freq == 0 :
            model_checkpoint_name = f"model_{i}.pth"
            model.save_pretrained(f"{checkpoint_path}/{model_checkpoint_name}")
            if config.save_ema:
                model_ema.save_pretrained(f"{checkpoint_path}/model_{i}_ema.pth")
            print("saved")

        if accelerator.is_main_process:
            if (i + 1) % config.render_freq == 0:
                print("=========== Rendering ==========")
                savepath=f"rendering/{config.env_id}/render_samples_{i}.png"
                os.makedirs(f"rendering/{config.env_id}", exist_ok=True)
                conditions = (s_t[:config.eval_batch_size],
                              h_t[:config.eval_batch_size],
                              actions[:config.eval_batch_size])

                x = generate_samples(config, conditions, model, dataset, noise_scheduler)
                observations = dataset.normalizer.unnormalize(x.cpu().numpy(), 'observations')
                renderer.composite(savepath, observations)


if __name__ == "__main__":
    config = tyro.cli(TrainingConfig)

    set_seed(config.seed)

    run_id = int(time.time())
    save_path = f"runs/{config.env_id}/{config.model_type}_{run_id}"
    os.makedirs(save_path, exist_ok=True)
    while os.path.exists(save_path):
        run_id = int(time.time())
        save_path = f"runs/{config.env_id}/{config.model_type}_{run_id}"
    
    dataset = SequenceDataset(config.env_id, horizon=config.horizon + config.history, normalizer="GaussianNormalizer", seed=config.seed)
    train_dataloader = cycle (torch.utils.data.DataLoader(
        dataset, batch_size=config.train_batch_size, num_workers=config.num_workers, shuffle=True, pin_memory=True
        ))

    net_args = {'sample_size': 65536, 'sample_rate': None, 'in_channels': dataset.observation_dim + dataset.action_dim, 'out_channels': dataset.observation_dim + dataset.action_dim, 'extra_in_channels': 0, 'time_embedding_type': 'positional', 'flip_sin_to_cos': False, 'use_timestep_embedding': True, 'freq_shift': 1, 'down_block_types': ['DownResnetBlock1D', 'DownResnetBlock1D', 'DownResnetBlock1D', 'DownResnetBlock1D'], 'up_block_types': ['UpResnetBlock1D', 'UpResnetBlock1D', 'UpResnetBlock1D'], 'mid_block_type': 'MidResTemporalBlock1D', 'out_block_type': 'OutConv1DBlock', 'block_out_channels': [32, 64, 128, 256], 'act_fn': 'mish', 'norm_num_groups': 8, 'layers_per_block': 1, 'downsample_each_block': False, '_use_default_values': ['sample_rate']}
    network = UNet1DModel(**net_args).to(device)
    if config.torch_compile:
        network = torch.compile(network)

    # create the schulduler 
    if config.use_original_config:
        scheduler = DDPMScheduler(num_train_timesteps=config.num_train_timesteps,
                                  beta_schedule="squaredcos_cap_v2",
                                  clip_sample=False,
                                  variance_type="fixed_small_log",
                                  prediction_type="sample",
                                  )
    else:
        scheduler = DDPMScheduler(num_train_timesteps=config.num_train_timesteps, beta_schedule="squaredcos_cap_v2")

    optimizer = torch.optim.AdamW(network.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    if config.cosine_warmup:
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=config.lr_warmup_steps,
            num_training_steps=config.n_train_steps,
        )
    else:
        lr_scheduler = get_constant_schedule(
            optimizer=optimizer
        )
    renderer = MuJoCoRenderer(config.env_id)

    # global_step = 0

    if config.wandb_track:
        wandb.init(
            config=config,
            name=str(run_id),
            project="diffusion_training",
            entity="pgm-diffusion"
        )

    args = (config, network, scheduler, optimizer, train_dataloader, lr_scheduler, renderer, save_path)
    
    train_loop(*args)
