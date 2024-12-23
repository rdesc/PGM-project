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
import numpy as np
import gym
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
    shape = (config.eval_batch_size, config.horizon, dataset.action_dim,)
    x = torch.randn(shape, device=device, generator=generator).to(device)
    s_t, h_t = conditioning
    if use_pipeline:
        for i, t in enumerate(scheduler.timesteps):
            timesteps = torch.full((config.eval_batch_size,), t, device=device, dtype=torch.long)
            model_input = scheduler.scale_model_input(x, t)

            s_t_expanded = s_t.unsqueeze(2).repeat(1, 1, config.horizon)  # batch_size x state_dim x horizon

            if config.history:
                h_t_expanded = h_t.repeat(1, 1, config.horizon)  # batch_size x (action_dim + state_dim) x horizon
                model_input = torch.cat([model_input.permute(0, 2, 1), s_t_expanded, h_t_expanded], dim=1)
            else:
                model_input = torch.cat([model_input.permute(0, 2, 1), s_t_expanded], dim=1)
            
            with torch.no_grad():
                noise_pred = model(model_input.permute(0, 2, 1), timesteps).sample
                noise_pred = noise_pred.permute(0, 2, 1) # needed to match model params to original
            x = scheduler.step(noise_pred, t, x).prev_sample

        # if config.use_conditioning_for_sampling:
        
    else:
        # sample random initial noise vector
        eta = 1.0 # noise factor for sampling reconstructed state

        # run the diffusion process
        s_t_expanded = s_t.unsqueeze(2).repeat(1, 1, config.horizon)  # batch_size x state_dim x horizon

        if config.history:
            h_t_expanded = h_t.repeat(1, 1, config.horizon)  # batch_size x (action_dim + state_dim) x horizon
            x = torch.cat([x.permute(0, 2, 1), s_t_expanded, h_t_expanded], dim=1).permute(0, 2, 1)
        else:
            x = torch.cat([x.permute(0, 2, 1), s_t_expanded], dim=1).permute(0, 2, 1)

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
            
            # if config.use_conditioning_for_sampling:

            # 4. apply conditions to the trajectory
            # obs_reconstruct_postcond = reset_x0(obs_reconstruct, conditions, action_dim)
            x = obs_reconstruct
    if obs_only:
        # for each action get the corresponding observation
        observations = []
        for i in range(config.eval_batch_size):
            curr_observations = []
            env = gym.make(config.env_id)
            env.reset()
            start_state = s_t[i].cpu().numpy()
            denorm_state = dataset.normalizer.unnormalize(start_state, 'observations')

            env.sim.data.qpos[1:] = denorm_state[:5]  # FIXME robot’s x-coordinate (rootx) not included... 
            env.sim.data.qvel[:] = denorm_state[5:]  # NOTE: hardcoded to Hopper

            for j in range(config.horizon):
                curr_action = x[i][j][:dataset.action_dim].cpu().numpy()
                denorm_action = dataset.normalizer.unnormalize(curr_action, 'actions')
                # execute action in environment
                next_observation, *_ = env.step(denorm_action)

                curr_observations.append(next_observation)

            observations.append(curr_observations)
        
        x = torch.tensor(np.array(observations))

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
    num_train_timesteps: int = 100  # set 
    n_train_steps: int = int(200e3)
    n_train_step_per_epoch: int = 10_000
    mixed_precision: str = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    horizon: int = 32
    seed: int = 0
    use_conditioning_for_sampling: bool = True
    checkpointing_freq: int = 100000
    wandb_track: bool = True
    num_workers: int = 1
    torch_compile: bool = True
    model_type: str = 'action_unet'  # 'action_unet', 'dynamics_unet'
    use_original_config: bool = True
    train_on_one_traj: bool = False
    use_grad_clip: bool = False # try training without gradient 
    weight_decay: float = 0.01 # try training with weight decay zero
    ema_decay: float = 0.995
    save_ema: bool = True
    update_ema_every: int = 10
    history: int = 0  # 0 or 1
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
    )
    if accelerator.is_main_process:
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
        batch = next(train_dataloader)
        trajectories = batch.trajectories.to(device)
        if config.train_on_one_traj:
            if one_trajectory is None:
                one_trajectory = torch.clone(trajectories[0][None, :])
            trajectories = one_trajectory

        trajectories = torch.permute(trajectories, (0,2,1) )  # shape (bs, state_dim + action_dim, horizon + history)
        actions = trajectories[:, :dataset.action_dim, config.history:]  # shape (bs, action_dim, horizon)

        # extract the history and the current state
        s_t = trajectories[:, dataset.action_dim:, config.history]  # shape (bs, state_dim)
        
        if config.history:
            h_t = trajectories[:, :, :config.history]  # shape (bs, state_dim + action_dim, history)
        
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

        # set the appropiate shape
        noisy_trajectories = noisy_trajectories[:, :dataset.action_dim, config.history:]  # shape (bs, action_dim, horizon)
        
        # add conditioning for the first state s_t and the history h_t
        s_t_expanded = s_t.unsqueeze(2).repeat(1, 1, config.horizon)  # batch_size x state_dim x horizon

        if config.history:
            h_t_expanded = h_t.repeat(1, 1, config.horizon)  # batch_size x (action_dim + state_dim) x horizon
            noisy_trajectories = torch.cat([noisy_trajectories, s_t_expanded, h_t_expanded], dim=1)
        else:
            noisy_trajectories = torch.cat([noisy_trajectories, s_t_expanded], dim=1)

        with accelerator.accumulate(model):
            # Predict the noise residual
            if config.use_original_config:
                sample_pred = model(noisy_trajectories, timesteps, return_dict=False)[0]
            # else:
            #     noise_pred = model(noisy_trajectories, timesteps, return_dict=False)[0]

            if config.use_original_config:
                # no need to take loss over entire sample_pred right? just actions
                loss = F.mse_loss(sample_pred[:, :dataset.action_dim], actions, reduction='none')
            # else:
            #     # loss = F.mse_loss(noise_pred, noise)
            #     loss = F.mse_loss(noise_pred, trajectories, reduction='none')

            # compute loss just for immediate action prediction, for logging
            a0_loss = loss[:, :dataset.action_dim, 0].mean().detach().item()
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

            logs = {"loss": weighted_loss.detach().item(), "a0_loss": a0_loss, "lr": lr_scheduler.get_last_lr()[0], "step": i}
            if config.wandb_track:
                wandb.log(logs, step=i)
            accelerator.log(logs, step=i)

        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if (i + 1) % config.checkpointing_freq == 0 :
            model_checkpoint_name = f"model_{i}.pth"
            model.save_pretrained(f"{checkpoint_path}/{model_checkpoint_name}")
            if config.save_ema:
                model_ema.save_pretrained(f"{checkpoint_path}/model_{i}_ema.pth")
            print("saved")

        if accelerator.is_main_process:
            if config.render_freq and (i + 1) % config.render_freq == 0:
                print("=========== Rendering ==========")
                savepath=f"rendering/{config.env_id}/render_samples_{i}.png"
                os.makedirs(f"rendering/{config.env_id}", exist_ok=True)
                if config.history:
                    conditions = (s_t[:config.eval_batch_size],
                                  h_t[:config.eval_batch_size])
                else:
                    conditions = (s_t[:config.eval_batch_size],
                                  None)

                x = generate_samples(config, conditions, model, dataset, noise_scheduler)
                observations = dataset.normalizer.unnormalize(x.cpu().numpy(), 'observations')
                renderer.composite(savepath, observations)


if __name__ == "__main__":
    config = tyro.cli(TrainingConfig)

    set_seed(config.seed)

    run_id = int(time.time())
    save_path = f"runs/{config.env_id}/{config.model_type}_{run_id}"
    while os.path.exists(save_path):
        run_id = int(time.time())
        save_path = f"runs/{config.env_id}/{config.model_type}_{run_id}"
    os.makedirs(save_path, exist_ok=True)
    config.output_dir = save_path

    dataset = SequenceDataset(config.env_id, horizon=config.horizon + config.history, normalizer="GaussianNormalizer", seed=config.seed)
    train_dataloader = cycle (torch.utils.data.DataLoader(
        dataset, batch_size=config.train_batch_size, num_workers=config.num_workers, shuffle=True, pin_memory=True
        ))

    net_args = {'sample_size': 65536, 'sample_rate': None, 'in_channels': dataset.action_dim + dataset.observation_dim + config.history * (dataset.action_dim + dataset.observation_dim), 'out_channels': dataset.action_dim + dataset.observation_dim + config.history * (dataset.action_dim + dataset.observation_dim), 'extra_in_channels': 0, 'time_embedding_type': 'positional', 'flip_sin_to_cos': False, 'use_timestep_embedding': True, 'freq_shift': 1, 'down_block_types': ['DownResnetBlock1D', 'DownResnetBlock1D', 'DownResnetBlock1D', 'DownResnetBlock1D'], 'up_block_types': ['UpResnetBlock1D', 'UpResnetBlock1D', 'UpResnetBlock1D'], 'mid_block_type': 'MidResTemporalBlock1D', 'out_block_type': 'OutConv1DBlock', 'block_out_channels': [32, 64, 128, 256], 'act_fn': 'mish', 'norm_num_groups': 8, 'layers_per_block': 1, 'downsample_each_block': False, '_use_default_values': ['sample_rate']}
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

    if config.wandb_track:
        wandb.init(
            config=config,
            name=str(run_id),
            project="diffusion_training",
            entity="pgm-diffusion"
        )

    args = (config, network, scheduler, optimizer, train_dataloader, lr_scheduler, renderer, save_path)
    
    train_loop(*args)
