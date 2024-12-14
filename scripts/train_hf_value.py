from dataclasses import dataclass
from tqdm import tqdm
from tqdm.auto import tqdm
from pathlib import Path

import os
import time

import torch
import torch.nn.functional as F

from accelerate import Accelerator
from huggingface_hub import HfFolder, Repository, whoami

from diffusers import  UNet1DModel, DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_cosine_schedule_with_warmup

from diffuser.utils.rendering import MuJoCoRenderer
from diffuser.datasets import ValueDataset
from diffuser.utils import set_seed
from transformer_1d import ValueTransformer

import tyro
import wandb

def cycle(dl):
    while True:
        for data in dl:
            yield data


@dataclass
class TrainingConfig:
    env_id: str = "hopper-medium-v2"
    train_batch_size: int = 64
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-4
    lr_warmup_steps: int = 500
    num_train_timesteps: int = 100
    n_train_steps: int = 200e3
    mixed_precision: str = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    # output_dir: str = "diffuser-value-hopperv2-32"  # the model name locally and on the HF Hub
    horizon: int = 32
    seed: int = 0
    use_ema: bool = True
    discount_factor: float = 0.997
    ema_decay: float = 0.995
    update_ema_steps: int = 10
    update_ema_start: int = 2000
    save_model_steps: int = 1e4
    num_workers: int = 1
    torch_compile: bool = True
    wandb_track: bool = True
    model_type: str = "value"
    arch_type: str = "unet"
    model_config_path: str = "bglick13/hopper-medium-v2-value-function-hor32"
    nheads: int = 4
    hidden_dim: int = 256
    num_layers: int = 5

def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler, renderer):
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with=None,
        project_dir=os.path.join(config.output_dir, "logs"),
    )

    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        accelerator.init_trackers("train_example")

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    train_dataloader = cycle(train_dataloader)
    if config.use_ema:
        ema = EMAModel(model.parameters(), 
                        decay=config.ema_decay,
                        model_cls=UNet1DModel if config.arch_type=="unet" else ValueTransformer,
                        model_config=value_network.config)
        ema.to(accelerator.device)

    # Now you train the model
    for i in tqdm(range(int(config.n_train_steps))):

        batch = next(train_dataloader)

        trajectories = batch.trajectories
        values = batch.values
        conditions = batch.conditions

        trajectories = torch.permute(trajectories, (0,2,1))

        # Sample noise to add to the images
        noise = torch.randn(trajectories.shape).to(accelerator.device)
        bs = trajectories.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps, (bs,), device=accelerator.device
        ).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectories = noise_scheduler.add_noise(trajectories, noise, timesteps)
        noisy_trajectories[:,dataset.action_dim:, 0] = conditions[0]

        with accelerator.accumulate(model):
            # Predict the noise residual
            values_pred = model(noisy_trajectories, timesteps, return_dict=False)[0]

            loss = F.mse_loss(values_pred, values)
            accelerator.backward(loss)

            # accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            if config.use_ema and i % config.update_ema_steps == 0:
                if i < config.update_ema_start: ## This is used in Diffuser repo
                    ## taken from the EMAModel file in HF 
                    parameters = model.parameters()
                    parameters = list(parameters)
                    ema.shadow_params = [p.clone().detach() for p in parameters]
                else:
                    ema.step(model.parameters())


            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": i}
            if config.wandb_track:
                wandb.log(logs, step=i)

        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            if (i + 1) % config.save_model_steps == 0:
                # print("Saving model....", "output_dir", config.output_dir, i+1, "loss", loss.detach().item()) 
                if config.use_ema:
                    accelerator.unwrap_model(model).save_pretrained(os.path.join(config.output_dir, "unet"), variant=str(i+1))
                    ema.store(model.parameters())
                    ema.copy_to(model.parameters())
                    accelerator.unwrap_model(model).save_pretrained(os.path.join(config.output_dir, "ema"), variant=str(i+1))
                    ema.restore(model.parameters())
                else:
                    accelerator.unwrap_model(model).save_pretrained(os.path.join(config.output_dir, "unet"), variant=str(i+1))



if __name__ == "__main__":
    config = tyro.cli(TrainingConfig)
    set_seed(config.seed)
    run_id = int(time.time())
    config.output_dir = f"runs/{config.model_type}_{run_id}"

    if config.wandb_track:
        wandb.init(
            config=config,
            name=config.output_dir,
            project="diffusion_training",
            entity="pgm-diffusion"
        )
        
    print("Discount factor:", config.discount_factor)
    dataset = ValueDataset(config.env_id, horizon=config.horizon, normalizer="GaussianNormalizer" , termination_penalty=-100, discount=config.discount_factor, seed=config.seed)
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, num_workers=config.num_workers, shuffle=True, pin_memory=True)

    assert config.arch_type in ['unet', 'transformer'], "Only unet and transformer are supported"
    
    if config.arch_type == 'unet':
        value_network_config = UNet1DModel.load_config(config.model_config_path, subfolder="value_function")
        print(value_network_config)
        value_network = UNet1DModel.from_config(value_network_config)
    else:
        nheads = config.nheads
        hidden_dim = config.hidden_dim
        num_layers = config.num_layers


        value_network_config = dict(
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
            num_positional_embeddings = config.horizon * 2 + 1,
            ff_inner_mult = 2,
            state_dim = dataset.observation_dim,
            action_dim = dataset.action_dim  ,
        )
        value_network = ValueTransformer(**value_network_config)

    if config.torch_compile:
        value_network = torch.compile(value_network)

    # create the schulduler 
    # create the schulduler 
    scheduler_config = DDPMScheduler.load_config(config.model_config_path, subfolder="scheduler")
    # below are kwargs to overwrite the config loaded from HF
    scheduler_config["num_train_timesteps"] = config.num_train_timesteps

    scheduler = DDPMScheduler.from_config(scheduler_config)
    optimizer = torch.optim.Adam(value_network.parameters(), lr=config.learning_rate)

    if config.wandb_track:
        wandb.config["model_config"] = value_network_config
        wandb.config["scheduler_config"] = scheduler_config
        wandb.config["optimizer"] = optimizer.__class__.__name__


    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=config.n_train_steps,
    )
    renderer = MuJoCoRenderer(config.env_id)
    args = (config, value_network, scheduler, optimizer, train_dataloader, lr_scheduler, renderer)

    train_loop(*args)
