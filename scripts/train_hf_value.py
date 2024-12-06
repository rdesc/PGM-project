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
    ema_decay: float = 0.995
    update_ema_steps: int = 10
    update_ema_start: int = 2000
    save_model_steps: int = 1e4
    num_workers: int = 1
    torch_compile: bool = True
    wandb_track: bool = True
    model_type: str = "value"
    model_config_path: str = "bglick13/hopper-medium-v2-value-function-hor32"



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
                        decay=0.995,
                        model_cls=UNet1DModel,
                        model_config=value_network.config)
        ema.to(accelerator.device)

    # Now you train the model
    for i in tqdm(range(int(config.n_train_steps))):

        batch = next(train_dataloader)

        trajectories = batch.trajectories
        values = batch.values
        conditions = batch.conditions

        trajectories = torch.permute(trajectories, (0,2,1) )

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
                accelerator.unwrap_model(model).save_pretrained(os.path.join(config.output_dir, "unet"), variant=str(i+1))
    if config.use_ema:
        ema.store(model.parameters())
        ema.copy_to(model.parameters())
        accelerator.unwrap_model(model).save_pretrained(os.path.join(config.output_dir, "ema"))
        ema.restore(model.parameters())



if __name__ == "__main__":
    config = tyro.cli(TrainingConfig)
    run_id = int(time.time())
    config.output_dir = f"{config.model_type}_{run_id}"

    if config.wandb_track:
        wandb.init(
            config=config,
            name=config.output_dir,
            project="diffusion_training",
            entity="pgm-diffusion"
        )
    
        
    dataset = ValueDataset(config.env_id, horizon=config.horizon, normalizer="GaussianNormalizer" , termination_penalty=-100)
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, num_workers=config.num_workers, shuffle=True, pin_memory=True)

    # net_args ={"in_channels": dataset.observation_dim + dataset.action_dim, 
    #            "down_block_types": ["DownResnetBlock1D", "DownResnetBlock1D", "DownResnetBlock1D", "DownResnetBlock1D", "DownResnetBlock1D"], 
    #            "up_block_types": [], "out_block_type": "ValueFunction", "mid_block_type": "ValueFunctionMidBlock1D", 
    #            "block_out_channels": [32, 64, 128, 256, 512], "layers_per_block": 1, "downsample_each_block": True, "sample_size": 65536, 
    #            "out_channels": dataset.observation_dim + dataset.action_dim, "extra_in_channels": 0, "time_embedding_type": "positional", 
    #            "use_timestep_embedding": True, "flip_sin_to_cos": False, "freq_shift": 1, "norm_num_groups": 8, "act_fn": "mish"}
    
    # if config.wandb_track:
    #     wandb.config["network_parameters"] = net_args

    # value_network = UNet1DModel(**net_args)
    value_network_config = UNet1DModel.load_config(config.model_config_path, subfolder="value_function")
    print(value_network_config)
    value_network = UNet1DModel.from_config(value_network_config)


    if config.torch_compile:
        value_network = torch.compile(value_network)

    # create the schulduler 

    # scheduler = DDPMScheduler(num_train_timesteps=config.num_train_timesteps,beta_schedule="squaredcos_cap_v2")
    scheduler_config = DDPMScheduler.load_config(config.model_config_path, subfolder="scheduler")
    print(scheduler_config)
    scheduler = DDPMScheduler.from_config(scheduler_config)


    if config.wandb_track:
        wandb.config["model_config"] = value_network_config
        wandb.config["scheduler_config"] = scheduler_config

    optimizer = torch.optim.Adam(value_network.parameters(), lr=config.learning_rate)

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=config.n_train_steps,
    )
    renderer = MuJoCoRenderer(config.env_id)
    args = (config, value_network, scheduler, optimizer, train_dataloader, lr_scheduler, renderer)

    train_loop(*args)