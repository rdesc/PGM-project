from dataclasses import dataclass
from tqdm import tqdm
from tqdm.auto import tqdm
from pathlib import Path

import os

import torch
import torch.nn.functional as F

from accelerate import Accelerator
from huggingface_hub import HfFolder, Repository, whoami

from diffusers import  UNet1DModel, DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_cosine_schedule_with_warmup

from diffuser.utils.rendering import MuJoCoRenderer
from diffuser.datasets import ValueDataset

def cycle(dl):
    while True:
        for data in dl:
            yield data

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device", device)

@dataclass
class TrainingConfig:
    env_id = "hopper-medium-v2"
    train_batch_size = 64
    gradient_accumulation_steps = 1
    learning_rate = 2e-4
    lr_warmup_steps = 500
    num_train_timesteps = 100
    n_train_steps= 200e3
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "diffuser-value-hopperv2-32"  # the model name locally and on the HF Hub
    horizon = 32
    seed = 0
    use_ema = True
    ema_decay = 0.995
    update_ema_steps = 10
    save_model_steps = 1e4



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
                ema.step(model.parameters())


        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            if (i + 1) % config.save_model_steps == 0:
                print("Saving model....", "output_dir", config.output_dir, i+1, "loss", loss.detach().item()) 
                accelerator.unwrap_model(model).save_pretrained(os.path.join(config.output_dir, "unet"), variant=str(i+1))
                if config.use_ema:
                    ema.store(model.parameters())
                    ema.copy_to(model.parameters())
                    accelerator.unwrap_model(model).save_pretrained(os.path.join(config.output_dir, "ema"), variant=str(i+1))
                    ema.restore(model.parameters())



if __name__ == "__main__":
    config = TrainingConfig()
    dataset = ValueDataset(config.env_id, horizon=config.horizon, normalizer="GaussianNormalizer" , termination_penalty=-100)
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, num_workers=1, shuffle=True, pin_memory=True)

    net_args ={"in_channels": dataset.observation_dim + dataset.action_dim, 
               "down_block_types": ["DownResnetBlock1D", "DownResnetBlock1D", "DownResnetBlock1D", "DownResnetBlock1D"], 
               "up_block_types": [], "out_block_type": "ValueFunction", "mid_block_type": "ValueFunctionMidBlock1D", 
               "block_out_channels": [32, 64, 128, 256], "layers_per_block": 1, "downsample_each_block": True, "sample_size": 65536, 
               "out_channels": dataset.observation_dim + dataset.action_dim, "extra_in_channels": 0, "time_embedding_type": "positional", 
               "use_timestep_embedding": True, "flip_sin_to_cos": False, "freq_shift": 1, "norm_num_groups": 8, "act_fn": "mish"}
    
    
    value_network = UNet1DModel(**net_args)

    # create the schulduler 

    scheduler = DDPMScheduler(num_train_timesteps=config.num_train_timesteps,beta_schedule="squaredcos_cap_v2")

    optimizer = torch.optim.AdamW(value_network.parameters(), lr=config.learning_rate)

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=config.n_train_steps,
    )
    renderer = MuJoCoRenderer(config.env_id)
    args = (config, value_network, scheduler, optimizer, train_dataloader, lr_scheduler, renderer)

    train_loop(*args)











    

