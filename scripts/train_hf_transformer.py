import os
# ## Ad mujoco path the LD_LIBRARY_PATH environment variable
# mujoco_path = "/home/mila/f/faisal.mohamed/.mujoco/mujoco200/bin"
# # Get the current LD_LIBRARY_PATH or initialize if not set
# current_ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
# # Add the new path if it's not already present
# if mujoco_path not in current_ld_library_path:
#     os.environ["LD_LIBRARY_PATH"] = f"{mujoco_path}:{current_ld_library_path}"
from dataclasses import dataclass, asdict
from diffuser.datasets import SequenceDataset
from diffusers import  DDPMScheduler
from transformer_1d import DiffuserTransformer
from tqdm import tqdm
from accelerate import Accelerator
from huggingface_hub import HfFolder, Repository, whoami
from diffusers.optimization import get_cosine_schedule_with_warmup, get_constant_schedule
from tqdm.auto import tqdm
from pathlib import Path
import torch
import torch.nn.functional as F
from diffusers import DDPMPipeline
import accelerate
from diffuser.utils.rendering import MuJoCoRenderer
from diffuser.utils import set_seed
import time
import yaml
import tyro
import wandb
import copy


# from  wonderwords import RandomWord

# def make_random_name():
#     r = RandomWord()
#     name = "-".join(
#         [r.word(word_min_length=3, word_max_length=7, include_parts_of_speech=["adjective"]),
#             r.word(word_min_length=5, word_max_length=7, include_parts_of_speech=["noun"])])
#     return name


def cycle(dl):
    while True:
        for data in dl:
            yield data

device = "cuda" if torch.cuda.is_available() else "cpu"


# def generate_samples(config, conditioning ,model, renderer, dataset, accelerator, scheduler, savepath ,n_samples=2, use_pipeline=False):
def generate_samples(config, conditioning ,model, dataset, scheduler, use_pipeline=False):
    generator = torch.Generator(device=device)
    shape = (config.eval_batch_size, config.horizon, dataset.observation_dim + dataset.action_dim,)
    x = torch.randn(shape, device=device, generator=generator).to(device)
    if use_pipeline:
        raise NotImplementedError
        for i, t in enumerate(scheduler.timesteps):
            timesteps = torch.full((config.eval_batch_size,), t, device=device, dtype=torch.long)
            model_input = scheduler.scale_model_input(x, t)
            if config.use_conditioning_for_sampling:
                # print('m', model_input.shape)
                # print(t)
                # print('c',conditioning.shape)
                model_input[:, 0, dataset.action_dim: ] = conditioning
            with torch.no_grad():
                noise_pred = model(model_input.permute(0, 2, 1), timesteps)
                noise_pred = noise_pred.permute(0, 2, 1) # needed to match model params to original
            x = scheduler.step(noise_pred, t, x).prev_sample
        if config.use_conditioning_for_sampling:
            x[:, 0, dataset.action_dim: ] = conditioning
        
    else:
        # sample random initial noise vector

        eta = 1.0 # noise factor for sampling reconstructed state

        # run the diffusion process
        # for i in tqdm.tqdm(reversed(range(num_inference_steps)), total=num_inference_steps):
        # import pdb; pdb.set_trace()
        if config.use_conditioning_for_sampling:
            x[:, 0, dataset.action_dim: ] = conditioning
        for i in tqdm(scheduler.timesteps):

            # create batch of timesteps to pass into model
            timesteps = torch.full((config.eval_batch_size,), i, device=device, dtype=torch.long)

            # 1. generate prediction from model
            with torch.no_grad():
                sample_actions = x[:,:,:dataset.action_dim]
                sample_states = x[:,:,dataset.action_dim:]
                pred_states, pred_actions = model(sample_states, sample_actions, timesteps)
                residual = torch.cat([pred_actions, pred_states], dim=-1)

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
                obs_reconstruct[:, 0, dataset.action_dim: ] = conditioning

            # 4. apply conditions to the trajectory
            # obs_reconstruct_postcond = reset_x0(obs_reconstruct, conditions, action_dim)
            x = obs_reconstruct
    x = x[:, : , dataset.action_dim:]
    return x

def render_samples(config, model, renderer, dataset, accelerator, noise_scheduler, savepath ,n_samples=2, use_pipeline=False, conditioning=None):
    x = generate_samples(config, conditioning ,model, dataset, noise_scheduler, use_pipeline=use_pipeline)
    # x = generate_samples(config, conditioning ,model, renderer, dataset, accelerator, noise_scheduler, savepath ,n_samples=n_samples, use_pipeline=use_pipeline)
    observations = dataset.normalizer.unnormalize(x.cpu().numpy(), 'observations')
    renderer.composite(savepath, observations)

def save_configs(configs, save_path):
    file_path = f"{save_path}/config.yaml"
    with open(file_path, 'w') as f:
        f.write(yaml.dump(asdict(configs)))



@dataclass
class TrainingConfig:
    env_id: str = "hopper-medium-v2"
    train_batch_size: int = 16
    eval_batch_size: int = 1  # how many images to sample during evaluation
    # num_epochs: int = 50
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    lr_warmup_steps: int = 500
    cosine_warmup: bool = True
    render_freq: int = 4200
    # save_model_epochs: int = 30
    num_train_timesteps: int = 100
    n_train_steps: int = int(200e3)
    n_train_step_per_epoch: int = 10_000
    mixed_precision: str = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    horizon: int = 128
    push_to_hub: bool = True  # whether to upload the saved model to the HF Hub
    hub_private_repo: bool = False
    overwrite_output_dir: bool = True  # overwrite the old model when re-running the notebook
    seed: int = 0
    use_sample_hf_for_render: bool= False
    add_training_conditioning: bool = True
    use_conditioning_for_sampling: bool = True
    checkpointing_freq: int = 20_000
    wandb_track: bool = True
    num_workers: int = 1
    torch_compile: bool = True
    model_type: str = 'diffusion_transformer'
    pred_noise: bool = False
    train_on_one_traj: bool = False
    use_grad_clip: bool = False # try training without gradient 
    weight_decay: float = 0.01 # try training with weight decay zero
    ema_decay: float = 0.995
    save_ema: bool = True
    action_weight: int = 1
    update_ema_every: int = 10
    nheads: int = 4
    hidden_dim: int = 256
    num_layers: int = 5

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
        # log_with="tensorboard",
        # project_dir=os.path.join(config.output_dir, "logs"),
    )
    if accelerator.is_main_process:
        # if config.push_to_hub:
            # repo_name = get_full_repo_name(Path(config.output_dir).name)
            # repo = Repository(config.output_dir, clone_from=repo_name)
        # if config.output_dir is not None:
        #     os.makedirs(config.output_dir, exist_ok=True)
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
        # trajectories = torch.permute(trajectories, (0,2,1) )
        # Sample noise to add to the images

        noise = torch.randn(trajectories.shape).to(trajectories.device)
        bs = trajectories.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps, (bs,), device=trajectories.device
        ).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectories = noise_scheduler.add_noise(trajectories, noise, timesteps)
        # condition on the first state
        if config.add_training_conditioning:
            noisy_trajectories[:, 0,  dataset.action_dim:] = trajectories[:, 0, dataset.action_dim:]

        noisy_states = noisy_trajectories[:,:, dataset.action_dim:]
        noisy_actions = noisy_trajectories[:,:, :dataset.action_dim]
        
        with accelerator.accumulate(model):
            (denoised_states, denoised_actions) = model(noisy_states, noisy_actions, timesteps)
            pred = torch.cat((denoised_actions, denoised_states), dim=-1)
            
            # condition on the first state
            if config.add_training_conditioning:
                if not config.pred_noise:
                    pred[:, 0, dataset.action_dim:] = trajectories[:, 0,  dataset.action_dim: ]
                else:
                    # we want the loss to cancel so we predict the noise not the trajectories
                    pred[:, 0, dataset.action_dim:] = noise[:, 0,  dataset.action_dim: ]

            if not config.pred_noise:
                # loss = F.mse_loss(pred, trajectories)
                loss = F.mse_loss(pred, trajectories, reduction='none')
            else:
                # loss = F.mse_loss(pred, noise)
                loss = F.mse_loss(pred, noise, reduction='none')
            loss_weights = torch.ones_like(trajectories)
            loss_weights[:, 0, :dataset.action_dim] = config.action_weight

            a0_loss = loss[:, 0, :dataset.action_dim,].mean().detach().item()
            weighted_loss = (loss * loss_weights).mean()
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
            logs = {"loss": weighted_loss.detach().item(), "a0_loss": a0_loss, "lr": lr_scheduler.get_last_lr()[0], "step": i}
            if config.wandb_track:
                wandb.log(logs, step=i)
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
        #     pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)
            if (i + 1) % config.render_freq == 0:
                print("=========== Rendering ==========")
                savepath=f"rendering/{config.env_id}/render_samples_{i}.png"
                os.makedirs(f"rendering/{config.env_id}", exist_ok=True)
                # conditioning = condition if config.use_conditioning_for_sampling else None
                if config.use_conditioning_for_sampling:
                    condition = trajectories[:config.eval_batch_size, 0,  dataset.action_dim:]
                else:
                    condition = None
                render_samples(config, model, renderer, dataset, accelerator,
                               noise_scheduler, savepath, config.eval_batch_size, use_pipeline=config.use_sample_hf_for_render, conditioning=condition)


if __name__ == "__main__":
    config = tyro.cli(TrainingConfig)
    set_seed(config.seed)

    dataset = SequenceDataset(config.env_id, horizon=config.horizon, normalizer="GaussianNormalizer", seed=config.seed)
    train_dataloader = cycle (torch.utils.data.DataLoader(
        dataset, batch_size=config.train_batch_size, num_workers=config.num_workers, shuffle=True, pin_memory=True
        ))

    # nheads = 8
    # hidden_dim = 1024
    # nheads = 4
    # hidden_dim = 512
    # nheads = 4
    # hidden_dim = 256
    nheads = config.nheads
    hidden_dim = config.hidden_dim
    num_layers = config.num_layers


    transformer_config = dict(
        num_attention_heads = nheads,
        attention_head_dim = hidden_dim // nheads,
        # num_attention_heads = 8,
        # attention_head_dim = 1024 // 8,
        num_layers = num_layers,
        dropout = 0.0,
        attention_bias= False,
        activation_fn = "geglu",
        num_embeds_ada_norm = config.num_train_timesteps,
        upcast_attention = False,
        norm_type = "ada_joker_norm_zero",  
        norm_elementwise_affine = True,
        norm_eps = 1e-5,
        attention_type = "default",
        interpolation_scale  = None,
        positional_embeddings = "sinusoidal",
        num_positional_embeddings = config.horizon * 2,
        ff_inner_mult = 2,
        state_dim = dataset.observation_dim,
        action_dim = dataset.action_dim  ,
    )

    network = DiffuserTransformer(**transformer_config)
    n_trainable = sum(p.numel() for p in network.parameters() if p.requires_grad)
    print('trainable params:', n_trainable)

    if config.torch_compile:
        network = torch.compile(network)
    
    # create the schulduler 
    scheduler = DDPMScheduler(
        num_train_timesteps=config.num_train_timesteps,
        beta_schedule="squaredcos_cap_v2", 
        clip_sample=False, 
        variance_type="fixed_small_log",
        prediction_type="sample" if not config.pred_noise else "epsilon",
    )


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
    run_id = int(time.time())
    save_path = f"runs/{config.env_id}/{run_id}"
    os.makedirs(save_path, exist_ok=True)
    while os.path.exists(save_path):
        run_id = int(time.time())
        save_path = f"runs/{config.env_id}/{run_id}"


    args = (config, network, scheduler, optimizer, train_dataloader, lr_scheduler, renderer, save_path)
    
    # name = f'{make_random_name()}-{str(run_id)}'
    name = str(run_id)
    
    if config.wandb_track:
        wandb.init(
            config=config,
            name=name,
            project="diffusion_training",
            entity="pgm-diffusion"
        )

    train_loop(*args)











    

