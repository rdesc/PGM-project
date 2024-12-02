import os 

import d4rl  # noqa

import numpy as np
import torch

import gym
import tqdm
from diffusers.experimental import ValueGuidedRLPipeline
from diffusers import  UNet1DModel
from diffuser.utils.rendering import MuJoCoRenderer
import argparse


def parse_args():
    parser = argparse.ArgumentParser("")
    parser.add_argument("--env_name", type=str,
                        default="hopper-medium-v2", help="Name of the environment")
    parser.add_argument("-f", "--file_name_render", type=str,
                        default=None)
    parser.add_argument("--pretrained_model", type=str, 
                        default="./diffuser-value-hopperv2-32", help="Path to the pretrained model")
    parser.add_argument("--variant", type=int,
                        default=200000, help="Variant of the model")
    parser.add_argument("--n_samples", type=int,
                        default=64, help="Number of samples")
    parser.add_argument("--horizon", type=int,
                        default=32, help="Planning horizon")
    parser.add_argument("-T", "--num_inference_steps", type=int,
                        default=1000, help="Number of inference steps")
    parser.add_argument("--n_guide_steps", type=int,
                        default=2, help="Number of guide steps")
    # parser.add_argument("--scale_grad_by_std", type=bool,
    #                     default=True, help="Scale gradient by standard deviation")
    parser.add_argument("--scale", type=float,
                        default=0.1, help="Scale factor for gradient in classifier guidance")
    # parser.add_argument("--eta", type=float,
    #                     default=0.0, help="Eta value")
    # parser.add_argument("--t_grad_cutoff", type=int,
    #                     default=2, help="Gradient cutoff")
    parser.add_argument("--device", type=str,
                        default="cuda", help="Device to use")
    parser.add_argument("--render_steps", type=int,
                        default=50, help="Number of steps for saving a render")
    return parser.parse_args()


if __name__ == "__main__":
    config = parse_args()

    env_name = config.env_name

    # check if file exists
    file_name_render = config.file_name_render if config.file_name_render else os.path.basename(config.pretrained_model) + "_render"
    if os.path.exists(file_name_render + ".mp4") or os.path.exists(file_name_render + ".png"):
        print(f"File {file_name_render} already exists. Exiting.")
        exit()

    env = gym.make(env_name)
    renderer = MuJoCoRenderer(env_name)

    device = "cpu" if not (torch.cuda.is_available() and config.device == "cuda") else "cuda"

    value_unet = UNet1DModel.from_pretrained(config.pretrained_model, use_safe_tensors=True, subfolder="unet", variant=str(config.variant)).to(device)

    pipeline = ValueGuidedRLPipeline.from_pretrained(
        "bglick13/hopper-medium-v2-value-function-hor32",
        env=env,
    ).to(device)
    pipeline.value_function = value_unet
    pipeline.register_modules(value_functions=pipeline.value_function)

    env.seed(0)
    obs = env.reset()
    total_reward = 0
    total_score = 0
    T = config.num_inference_steps
    rollout = [obs.copy()]

    try:
        for t in tqdm.tqdm(range(T)):
            # call the policy
            denorm_actions = pipeline(obs,
                                      batch_size=config.n_samples,
                                      planning_horizon=config.horizon,
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
        
            if t % config.render_steps == 0: 
                renderer.render_rollout(f"./{file_name_render}_{t}.mp4", np.array(rollout))
                renderer.composite(f"./{file_name_render}_{t}.png", np.array(rollout)[None])

    except KeyboardInterrupt:
        pass

    print(f"Total reward: {total_reward}")