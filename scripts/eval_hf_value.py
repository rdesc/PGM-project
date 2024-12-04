import os 
from dataclasses import dataclass, asdict

# import d4rl  # noqa

import numpy as np
import torch

import gym
import tqdm
from diffusers.experimental import ValueGuidedRLPipeline
from diffusers import  UNet1DModel, DDPMScheduler, DDPMPipeline
from diffuser.utils.rendering import MuJoCoRenderer
from bc_d4rl import show_sample
import tyro

@dataclass
class TrainingConfig:
    env_name: str = "hopper-medium-v2"
    file_name_render: str = None
    batch_size: int = 64
    planning_horizon: int = 32
    max_episode_length: int = 1000
    n_guide_steps: int = 2
    scale: float = 0.1
    num_train_timesteps: int = 100
    render_steps: int = 50
    pretrained_value_model: str = None
    pretrained_diff_model: str = None
    checkpoint_diff_model: int = None
    checkpoint_value_model: int = None
    runid_diff_model: int = None
    hf_repo: str = "bglick13/hopper-medium-v2-value-function-hor32"
    use_ema: bool = True
    torch_compile: bool = True



if __name__ == "__main__":
    config = tyro.cli(TrainingConfig)

    print("config grad_scale", config.scale)
    env_name = config.env_name

    # check if file exists
    file_name_render = config.file_name_render if config.file_name_render else os.path.basename(config.pretrained_value_model) + "_render"
    if os.path.exists(file_name_render + ".mp4") or os.path.exists(file_name_render + ".png"):
        print(f"File {file_name_render} already exists. Exiting.")
        exit()

    env = gym.make(env_name)
    renderer = MuJoCoRenderer(env_name)

    device = "cpu" if not torch.cuda.is_available() else "cuda"

    if not config.pretrained_value_model is None:
        print("Loading value model from ", config.pretrained_value_model, config.checkpoint_value_model, "use-ema:", config.use_ema)
        value_function = UNet1DModel.from_pretrained(config.pretrained_value_model, use_safe_tensors=True, 
                                                 subfolder="ema" if config.use_ema else "unet", variant=str(config.checkpoint_value_model)).to(device)

    else:
        print("Loading value function from ", config.hf_repo)
        value_function = UNet1DModel.from_pretrained(config.hf_repo, subfolder="value_function", use_safe_tensors=False)

    if not config.pretrained_diff_model is None:
        print("Loading diffusion model from ", config.pretrained_diff_model, config.checkpoint_diff_model)

        pretrained_diff_path = os.path.join(config.pretrained_diff_model, str(config.runid_diff_model))
        unet = UNet1DModel.from_pretrained(os.path.join(pretrained_diff_path, "checkpoints/model_{}.pth".format(config.checkpoint_diff_model)))
        scheduler = DDPMScheduler.from_pretrained(pretrained_diff_path)
        print("num train timesteps", scheduler.num_train_timesteps)
        
    else:
        print("Loading diffusion model from ", config.hf_repo)
        unet = UNet1DModel.from_pretrained(config.hf_repo, subfolder="unet", use_safe_tensors=False)
        scheduler = DDPMScheduler.from_pretrained(config.hf_repo, subfolder="scheduler")
    
    if config.torch_compile:
        value_function = torch.compile(value_function)
        unet = torch.compile(unet)
        
    pipeline = ValueGuidedRLPipeline(value_function=value_function, unet=unet, scheduler=scheduler, env=env).to(device)

    obs = env.reset()
    total_reward = 0
    total_score = 0
    rollout = [obs.copy()]

    try:
        for t in tqdm.tqdm(range(config.max_episode_length)):
            # call the policy
            denorm_actions = pipeline(obs,
                                      batch_size=config.batch_size,
                                      planning_horizon=config.planning_horizon,
                                      n_guide_steps=config.n_guide_steps,
                                      scale=config.scale)

            # execute action in environment
            next_observation, reward, terminal, _ = env.step(denorm_actions)
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

    print(f"Total reward: {total_reward}")