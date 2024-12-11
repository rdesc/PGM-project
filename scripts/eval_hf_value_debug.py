import os 
from dataclasses import dataclass, asdict
from typing import Optional

# import d4rl  # noqa

import numpy as np
import torch
from diffuser.datasets import SequenceDataset

import gym
import tqdm
import time
# from diffusers.experimental import ValueGuidedRLPipeline
from value_guided_sampling import ValueGuidedRLPipeline
from diffusers import  UNet1DModel, DDPMScheduler, DDPMPipeline, DDIMScheduler
from diffuser.utils.rendering import MuJoCoRenderer
from diffuser.utils import set_seed
from bc_d4rl import show_sample
import tyro

def generate_samples_eval(config, conditioning ,model, dataset, scheduler, batch_size, horizon):
    generator = torch.Generator(device=device)
    shape = (batch_size, horizon, dataset.observation_dim + dataset.action_dim,)
    x = torch.randn(shape, device=device, generator=generator).to(device)
    for i, t in enumerate(scheduler.timesteps):
        timesteps = torch.full((batch_size,), t, device=device, dtype=torch.long)
        model_input = scheduler.scale_model_input(x, t)
        if conditioning is not None:
            # print('m', model_input.shape)
            # print(t)
            # print('c',conditioning.shape)
            model_input[:, 0, dataset.action_dim: ] = conditioning
        with torch.no_grad():
            noise_pred = model(model_input.permute(0, 2, 1), timesteps).sample
            noise_pred = noise_pred.permute(0, 2, 1) # needed to match model params to original
        if config.ddim:
            x = scheduler.step(noise_pred, t, x, eta=config.eta).prev_sample
        else:
            x = scheduler.step(noise_pred, t, x).prev_sample
    if conditioning is not None:
        x[:, 0, dataset.action_dim: ] = conditioning
        
    return x


@dataclass
class TrainingConfig:
    env_name: str = "hopper-medium-v2"
    """Name of the environment"""
    file_name_render: Optional[str] = None
    batch_size: int = 64  # the number of samples to generate, selects the best action
    planning_horizon: int = 32
    max_episode_length: int = 1000
    n_guide_steps: int = 2
    scale: float = 0.1
    num_inference_steps: int = 20  # this needs to be <= num_train_timesteps used during training
    render_steps: int = 50
    pretrained_value_model: Optional[str] = None
    pretrained_diff_model: Optional[str] = None
    checkpoint_value_model: Optional[str] = None
    checkpoint_diff_model: Optional[str] = None
    runid_diff_model: Optional[str] = None
    hf_repo: str = "bglick13/hopper-medium-v2-value-function-hor32"
    use_ema: bool = True
    torch_compile: bool = True
    seed: int = 0
    ddim: bool = False
    eta: float = 0.0


if __name__ == "__main__":
    config = tyro.cli(TrainingConfig)

    set_seed(config.seed)

    print("config grad_scale", config.scale)
    env_name = config.env_name

    # check if file exists
    file_name_render = config.file_name_render if config.file_name_render else os.path.basename(config.pretrained_value_model or config.hf_repo) + "_render"
    if os.path.exists(file_name_render + ".mp4") or os.path.exists(file_name_render + ".png"):
        print(f"File {file_name_render} already exists. Exiting.")
        exit()

    # env = gym.make(env_name)
    dataset = SequenceDataset(config.env_name, horizon=config.planning_horizon, normalizer="GaussianNormalizer", seed=config.seed)
    env = dataset.env
    env.seed(config.seed)
    renderer = MuJoCoRenderer(env_name)
    device = "cpu" if not torch.cuda.is_available() else "cuda"

    if not config.pretrained_value_model is None:
        print("Loading value model from ", config.pretrained_value_model, config.checkpoint_value_model, "use-ema:", config.use_ema)
        # value_function = UNet1DModel.from_pretrained(config.pretrained_value_model, use_safe_tensors=True, 
        #                                          subfolder="ema" if config.use_ema else "unet", variant= None if config.use_ema else str(config.checkpoint_value_model)).to(device)
        value_function = UNet1DModel.from_pretrained(config.pretrained_value_model, use_safe_tensors=True, 
                                            subfolder="ema" if config.use_ema else "unet", variant= str(config.checkpoint_value_model)).to(device)

    else:
        print("Loading value function from ", config.hf_repo)
        value_function = UNet1DModel.from_pretrained(config.hf_repo, subfolder="value_function", use_safe_tensors=False)

    if not config.pretrained_diff_model is None:
        pretrained_diff_path = os.path.join(config.pretrained_diff_model, str(config.runid_diff_model))
        print("Loading diffusion model from ", pretrained_diff_path, config.checkpoint_diff_model)

        unet = UNet1DModel.from_pretrained(os.path.join(pretrained_diff_path, "checkpoints/model_{}.pth".format(config.checkpoint_diff_model)))
        
        if config.ddim:
            print("using ddim")
            scheduler = DDIMScheduler.from_pretrained(
                pretrained_diff_path,
            ) #  when we call step/forward defaults to eta=0.0 , deterministic
        else:
            print("using ddpm")
            scheduler = DDPMScheduler.from_pretrained(pretrained_diff_path)

        
    else:
        print("Loading diffusion model from ", config.hf_repo)
        unet = UNet1DModel.from_pretrained(config.hf_repo, subfolder="unet", use_safe_tensors=False)
        scheduler = DDPMScheduler.from_pretrained(config.hf_repo, subfolder="scheduler",
                                                  # below are kwargs to overwrite the config loaded from the pretrained model
                                                  )
    scheduler.set_timesteps(config.num_inference_steps)
    print("num train timesteps", scheduler.num_train_timesteps, "num inference timesteps", scheduler.num_inference_steps)

    if config.torch_compile:
        value_function = torch.compile(value_function)
        unet = torch.compile(unet)
        
    pipeline = ValueGuidedRLPipeline(value_function=value_function, unet=unet, scheduler=scheduler, env=env).to(device)

    obs = env.reset()
    total_reward = 0
    total_score = 0
    rollout = [obs.copy()]
    print('starting timing')
    start_time = time.time()
    with torch.no_grad():
        try:
            for t in tqdm.tqdm(range(config.max_episode_length)):
                # call the policy
                # denorm_actions = pipeline(obs,
                #                           batch_size=config.batch_size,
                #                           planning_horizon=config.planning_horizon,
                #                           n_guide_steps=config.n_guide_steps,
                #                           scale=config.scale)
                # normalize observation
                norm_observation = dataset.normalizer.normalize(obs, 'observations')
                conditions = torch.tensor(norm_observation, device=device)
                conditions = conditions.repeat((config.batch_size, 1))
                samples = generate_samples_eval(config, conditions, unet, dataset, scheduler, config.batch_size, config.planning_horizon)
                # get values of samples
                timesteps = torch.full((config.batch_size,), 0, device=unet.device, dtype=torch.long)
                values = value_function(samples.permute(0, 2, 1), timesteps).sample
                sorted_idx = values.argsort(0, descending=True).squeeze()
                sorted_samples = samples[sorted_idx]
                # extract actions
                normed_actions = sorted_samples[:, 0, :dataset.action_dim ]
                best_action = normed_actions[0].cpu().numpy()
                denorm_action = dataset.normalizer.unnormalize(best_action, 'actions')

                # execute action in environment
                next_observation, reward, terminal, _ = env.step(denorm_action)
                score = env.get_normalized_score(total_reward)
                # update return
                total_reward += reward
                print(
                    f"Step: {t}, Reward: {reward}, Total Reward: {total_reward}, Score: {score}"
                )

                # save observations for rendering
                rollout.append(next_observation.copy())

                obs = next_observation
            
                if (t+1) % config.render_steps == 0: 
                    show_sample(renderer, [rollout], filename=f"{file_name_render}.mp4", savebase="./renders")
                    renderer.composite(f"./renders/{file_name_render}.png", np.array(rollout)[None])

        except KeyboardInterrupt:
            pass
    total_time =  time.time() - start_time
    print('eval time, includes some compilation', total_time)
    print(f"Total reward: {total_reward}, Score: {env.get_normalized_score(total_reward)}")