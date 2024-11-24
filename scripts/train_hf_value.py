from dataclasses import dataclass
from diffuser.datasets import ValueDataset
from diffusers import  UNet1DModel, DDPMScheduler
from tqdm import tqdm
from accelerate import Accelerator
from huggingface_hub import HfFolder, Repository, whoami
from tqdm.auto import tqdm
from pathlib import Path
import os
import torch
import torch.nn.functional as F
from diffusers import DDPMPipeline
from diffuser.utils.rendering import MuJoCoRenderer

def cycle(dl):
    while True:
        for data in dl:
            yield data

device = "cuda" if torch.cuda.is_available() else "cpu"



def render_samples(config, model, renderer, dataset, accelerator, noise_scheduler, savepath ,n_samples=2, use_pipeline=False):
    if use_pipeline:
        pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)
        x = pipeline(
                batch_size=config.eval_batch_size,
                generator=torch.manual_seed(config.seed),
                 output_type="numpy"
            ).images
        
    else:
        generator = torch.Generator(device=device)
        # sample random initial noise vector
        shape = (config.eval_batch_size, config.horizon, dataset.observation_dim + dataset.action_dim,)
        x = torch.randn(shape, device=device, generator=generator).to(device)

        eta = 1.0 # noise factor for sampling reconstructed state

        # run the diffusion process
        # for i in tqdm.tqdm(reversed(range(num_inference_steps)), total=num_inference_steps):
        for i in tqdm(scheduler.timesteps):

            # create batch of timesteps to pass into model
            timesteps = torch.full((config.eval_batch_size,), i, device=device, dtype=torch.long)

            # 1. generate prediction from model
            with torch.no_grad():
                residual = network(x.permute(0, 2, 1), timesteps).sample
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

            # 4. apply conditions to the trajectory
            # obs_reconstruct_postcond = reset_x0(obs_reconstruct, conditions, action_dim)
            x = obs_reconstruct
    x = x[:, : , dataset.action_dim:]
    # import pdb; pdb.set_trace()
    observations = dataset.normalizer.unnormalize(x.cpu().numpy(), 'observations')
    renderer.composite(savepath, observations)




@dataclass
class TrainingConfig:
    env_id = "hopper-medium-expert-v2"
    train_batch_size = 128
    eval_batch_size = 1  # how many images to sample during evaluation
    num_epochs = 50
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 10
    render_freq = 200e3 +1
    save_model_epochs = 30
    num_train_timesteps = 100
    n_train_steps= 200e3
    n_train_step_per_epoch = 10_000
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "diffuser-hopperv2-32"  # the model name locally and on the HF Hub
    horizon = 32
    push_to_hub = False  # whether to upload the saved model to the HF Hub
    hub_private_repo = False
    overwrite_output_dir = False  # overwrite the old model when re-running the notebook
    seed = 0
    use_pipeline_for_render= False


def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler, renderer):
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

    # global_step = 0

    # Now you train the model
    for i in tqdm(range(int(config.n_train_steps))):
        # progress_bar = tqdm(total=int(config.n_train_steps), disable=not accelerator.is_local_main_process)
        # progress_bar.set_description(f"Epoch {i}")

        batch = next(train_dataloader)
        trajectories = batch.trajectories.to(device)
        trajectories = torch.permute(trajectories, (0,2,1) )
        values = batch.values.to(device)
        # print(trajectories.shape, values.shape)
        # import pdb; pdb.set_trace()
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

        with accelerator.accumulate(model):
            # Predict the noise residual
            values_pred = model(noisy_trajectories, timesteps, return_dict=False)[0]
            # print(values_pred.shape)
            # exit(0)
            loss = F.mse_loss(values_pred, values)
            accelerator.backward(loss)

            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": i}
            # progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=i)
            # global_step += 1

        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)

            if (i + 1) % config.render_freq == 0:
                print("=========== Rendering ==========")
                savepath=f"rendering/{config.env_id}/render_samples_{i}.png"
                os.makedirs(f"rendering/{config.env_id}", exist_ok=True)
                render_samples(config, model, renderer, dataset, accelerator,
                               noise_scheduler, savepath, config.eval_batch_size, use_pipeline=config.use_pipeline_for_render)

            if (i + 1) % 10_000 == 0:
                print("Saving model....", "output_dir", config.output_dir, i+1, "loss", loss) 
                pipeline.save_pretrained(config.output_dir, variant=str(i+1))


if __name__ == "__main__":
    config = TrainingConfig()
    # env_id = 
    dataset = ValueDataset(config.env_id, horizon=config.horizon, normalizer="GaussianNormalizer")
    train_dataloader = cycle(torch.utils.data.DataLoader(
        dataset, batch_size=config.train_batch_size, num_workers=1, shuffle=True, pin_memory=True
        ))
    n_epochs = config.n_train_steps // config.n_train_step_per_epoch
    config.n_epochs = n_epochs
    # batch = next(train_dataloader)
    # import pdb; pdb.set_trace()
    # print(batch)
    # import pdb; pdb.set_trace()
    # network = UNet1DModel(
        # in_channels=dataset.observation_dim + dataset.action_dim,  # the number of input channels, 3 for RGB images
    #     out_channels=dataset.observation_dim + dataset.action_dim,  # the number of output channels
    #     layers_per_block=1,  # how many ResNet layers to use per UNet block
    #     block_out_channels=(32, 64, 128, 256),  # the number of output channels for each UNet block
    #     down_block_types= ['DownResnetBlock1D', 'DownResnetBlock1D', 'DownResnetBlock1D', 'DownResnetBlock1D'],
    #     up_block_types= ['UpResnetBlock1D', 'UpResnetBlock1D', 'UpResnetBlock1D'],
    #     mid_block_type= 'MidResTemporalBlock1D',
    #     time_embedding_type="positional",
    #     flip_sin_to_cos=False,
    #     use_timestep_embedding=True, 
    #     freq_shift=1,
    #     act_fn="mish",
        
    # )
    net_args ={"in_channels": dataset.observation_dim + dataset.action_dim, "down_block_types": ["DownResnetBlock1D", "DownResnetBlock1D", "DownResnetBlock1D", "DownResnetBlock1D"], "up_block_types": [], "out_block_type": "ValueFunction", "mid_block_type": "ValueFunctionMidBlock1D", "block_out_channels": [32, 64, 128, 256], "layers_per_block": 1, "downsample_each_block": True, "sample_size": 65536, "out_channels": dataset.observation_dim + dataset.action_dim, "extra_in_channels": 0, "time_embedding_type": "positional", "use_timestep_embedding": True, "flip_sin_to_cos": False, "freq_shift": 1, "norm_num_groups": 8, "act_fn": "mish"}
    value_network = UNet1DModel(**net_args).to(device)
    # print(value_network)
    # import pdb; pdb.set_trace()
    # trajetory = next(train_dataloader).trajectories[0].unsqueeze(0).to(device)
    # sample random initial noise vector
    # output = network(trajetory, timestep=0)


    # create the schulduler 

    scheduler = DDPMScheduler(num_train_timesteps=config.num_train_timesteps,beta_schedule="squaredcos_cap_v2")

    # obs = obs[None].repeat(n_samples, axis=0)
    # conditions = {
    #     0: to_torch(obs, device=DEVICE)
    # }

    # constants for inference
    # batch_size = len(conditions[0])
    # shape = (batch_size, horizon, state_dim+action_dim)

    # create optimizer and lr scheduler
    from diffusers.optimization import get_cosine_schedule_with_warmup

    optimizer = torch.optim.AdamW(value_network.parameters(), lr=config.learning_rate)
    # import pdb; pdb.set_trace()
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=config.n_train_steps,
    )
    renderer = MuJoCoRenderer(config.env_id)
    args = (config, value_network, scheduler, optimizer, train_dataloader, lr_scheduler, renderer)

    train_loop(*args)











    

