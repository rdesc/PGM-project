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
from transformer_1d import ActionProposalTransformer
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

device = "cuda" if torch.cuda.is_available() else "cpu"

def cycle(dl):
    while True:
        for data in dl:
            yield data



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
    model_type: str = 'action_transformer'
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
    grad_clip_val: float = 5.0

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

        actions = trajectories[:,:, :dataset.action_dim]
        initial_state = trajectories[:,0,dataset.action_dim:]


        # Sample noise to add to the images
        action_noise = torch.randn(actions.shape).to(actions.device)
        bs = actions.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps, (bs,), device=actions.device
        ).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_actions = noise_scheduler.add_noise(actions, action_noise, timesteps)

        with accelerator.accumulate(model):
            pred_actions = model(initial_state, noisy_actions, timesteps)

            if not config.pred_noise:
                loss = F.mse_loss(pred_actions, actions, reduction='none')
            else:
                loss = F.mse_loss(pred_actions, action_noise, reduction='none')

            a0_loss = loss[:, 0].mean()
            loss = loss.mean()
            accelerator.backward(loss)

            if config.use_grad_clip:
                accelerator.clip_grad_norm_(model.parameters(), config.grad_clip_val)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # update EMA
            if (i + 1) % config.update_ema_every == 0:
                update_ema(model_ema, model, config.ema_decay)

            # progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "a0_loss": a0_loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": i}
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
        num_positional_embeddings = config.horizon + 1,
        ff_inner_mult = 2,
        state_dim = dataset.observation_dim,
        action_dim = dataset.action_dim  ,
    )

    network = ActionProposalTransformer(**transformer_config)
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











    

