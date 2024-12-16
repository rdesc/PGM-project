import os 
import json
import time
from dataclasses import dataclass, asdict
from typing import Optional

# import d4rl  # noqa

import numpy as np
import torch

import gym
import tqdm
# from diffusers.experimental import ValueGuidedRLPipeline
from value_guided_sampling import ValueGuidedRLPipeline
from diffusers import  UNet1DModel, DDPMScheduler, DDPMPipeline
from transformer_1d import DiffuserTransformer, DiffuserTransformerPolicy
from diffuser.utils.rendering import MuJoCoRenderer
from diffuser.utils import set_seed
from diffuser.datasets import ValueDataset

from bc_d4rl import show_sample
import tyro
import wandb


MAPPING_DICT = {
    DiffuserTransformer: DiffuserTransformerPolicy,
}


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
    num_inference_steps: int = 100  # this needs to be <= num_train_timesteps used during training
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
    wandb_track: bool = True
    render: bool = True
    n_episodes: int = 1


if __name__ == "__main__":
    device = "cpu" if not torch.cuda.is_available() else "cuda"
    config = tyro.cli(TrainingConfig)
    run_id = int(time.time())

    set_seed(config.seed)


    print("Config:", config)
    env_name = config.env_name

    # check if file exists
    file_name_render = config.file_name_render if config.file_name_render else os.path.basename(config.pretrained_value_model or config.hf_repo) + "_render"
    if os.path.exists(file_name_render + ".mp4") or os.path.exists(file_name_render + ".png"):
        print(f"File {file_name_render} already exists. Exiting.")
        exit()

    dataset = ValueDataset(env_name, horizon=config.planning_horizon, normalizer="GaussianNormalizer" , termination_penalty=-100, discount=0.997, seed=config.seed)
    env = dataset.env
    env.seed(config.seed)

    if config.render:
        renderer = MuJoCoRenderer(env_name)


    if not config.pretrained_value_model is None:
        print("Loading value model from ", config.pretrained_value_model, config.checkpoint_value_model, "use-ema:", config.use_ema)
        value_function = UNet1DModel.from_pretrained(config.pretrained_value_model, use_safe_tensors=True, 
                                                 subfolder="ema" if config.use_ema else "unet", variant=str(config.checkpoint_value_model)).to(device)

    else:
        print("Loading value function from ", config.hf_repo)
        value_function = UNet1DModel.from_pretrained(config.hf_repo, subfolder="value_function", use_safe_tensors=False)

    if not config.pretrained_diff_model is None:
        if config.use_ema:
            model_diff_path = os.path.join(config.pretrained_diff_model, "checkpoints/model_{}_ema.pth".format(config.checkpoint_diff_model))
        else:
            model_diff_path = os.path.join(config.pretrained_diff_model, "checkpoints/model_{}.pth".format(config.checkpoint_diff_model))

        print("Loading diffusion model from ", model_diff_path)

        with open(os.path.join(model_diff_path, "config.json"), "rb") as f:
            model_config = json.load(f)

        class_str = model_config["_class_name"]
        class_name = eval(class_str)
        print(class_name)
        # diffusion_model = UNet1DModel.from_pretrained(model_diff_path)
        diffusion_model = class_name.from_pretrained(model_diff_path).to(device)
        
        if class_name in MAPPING_DICT:
            print("Mapping class")
            diffusion_model = MAPPING_DICT[class_name](diffusion_model)
        
        scheduler = DDPMScheduler.from_pretrained(config.pretrained_diff_model)
    else:
        print("Loading diffusion model from ", config.hf_repo)
        diffusion_model = UNet1DModel.from_pretrained(config.hf_repo, subfolder="unet", use_safe_tensors=False)
        scheduler = DDPMScheduler.from_pretrained(config.hf_repo, subfolder="scheduler")
    scheduler.set_timesteps(config.num_inference_steps)
    print("num train timesteps", scheduler.num_train_timesteps, "num inference timesteps", scheduler.num_inference_steps)

    if config.torch_compile:
        value_function = torch.compile(value_function)
        diffusion_model = torch.compile(diffusion_model)
        
    pipeline = ValueGuidedRLPipeline(value_function=value_function, unet=diffusion_model, scheduler=scheduler, env=env).to(device)

    config.model_type = 'diffuser'
    config.model_arch = class_str
    if config.wandb_track:
        wandb.init(
            config=config,
            name=str(run_id),
            project="diffusion_testing",
            entity="pgm-diffusion"
        )


    ep_returns = []
    ep_scores = []

    if config.n_episodes > 1:
        seeds = np.arange(config.n_episodes, dtype=int)
    else:
        seeds = [config.seed]


    for seed in seeds:
        set_seed(int(seed))
        env.seed(int(seed))
        obs = env.reset()
        total_reward = 0
        total_score = 0
        rollout = [obs.copy()]

        image = None
        for t in tqdm.tqdm(range(config.max_episode_length)):
            # call the policy
            denorm_actions = pipeline(obs,
                                    batch_size=config.batch_size,
                                    planning_horizon=config.planning_horizon,
                                    n_guide_steps=config.n_guide_steps,
                                    scale=config.scale)

            # execute action in environment
            next_observation, reward, terminal, _ = env.step(denorm_actions)
            # update return
            total_reward += reward

            # print(
            #     f"Step: {t}, Reward: {reward}, Total Reward: {total_reward}, Score: {score}"
            # )
            # save observations for rendering
            rollout.append(next_observation.copy())

            obs = next_observation

            if terminal:
                if config.render:
                    show_sample(renderer, [rollout], filename=f"{file_name_render}.mp4", savebase="./renders")
                    image = renderer.composite(f"./renders/{file_name_render}.png", np.array(rollout)[None])  
                break
            if config.render and (t+1) % config.render_steps == 0: 
                show_sample(renderer, [rollout], filename=f"{file_name_render}.mp4", savebase="./renders")
                image = renderer.composite(f"./renders/{file_name_render}.png", np.array(rollout)[None])

        normalized_score = env.get_normalized_score(total_reward)
        ep_returns.append(total_reward)
        ep_scores.append(normalized_score)

        print(f"Total reward: {total_reward}, Score: {env.get_normalized_score(total_reward)}")
        if config.wandb_track:
            logs = {"score": normalized_score, "total_reward":total_reward, 'seed': seed}
            import pdb; pdb.set_trace()
            if image is not None:
                logs['image'] = wandb.Image(image, caption=f"composite {seed}", file_type="png")
            wandb.log(logs)
    if config.wandb_track:
        wandb.summary["avg_return"] = np.mean(ep_returns)
        wandb.summary["avg_return"] = np.std(ep_returns)
        wandb.summary["avg_score"] = np.mean(ep_scores)
        wandb.summary["std_score"] = np.std(ep_scores) 