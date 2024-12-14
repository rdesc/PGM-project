import os 
import json
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import torch

import gym
import tqdm
from diffusers import DDPMScheduler, DDIMScheduler
from transformer_1d import ValueTransformer, DiffuserTransformerPolicy, ActionProposalTransformer, DynamicsTransformer
from diffuser.utils.rendering import MuJoCoRenderer
from diffuser.utils import set_seed
from bc_d4rl import show_sample
import tyro
from diffuser.datasets import ValueDataset

def generate_samples_dyn(observation, action_samples, model, scheduler, dataset, config):
    batch_size = config.batch_size
    horizon = config.planning_horizon
    generator = torch.Generator(device=device)
    state_samples = torch.randn((batch_size, horizon, dataset.observation_dim), device=device, generator=generator).to(device)
    for i, t in enumerate(scheduler.timesteps):
        timesteps = torch.full((batch_size,), t, device=device, dtype=torch.long)
        scaled_state_samples = scheduler.scale_model_input(state_samples, t)
        scaled_state_samples[:, 0] = observation
        with torch.no_grad():
            state_pred = model(scaled_state_samples, action_samples, timesteps).sample

        if config.ddim:
            state_samples = scheduler.step(state_pred, t, state_samples, eta=config.eta).prev_sample
        else:
            state_samples = scheduler.step(state_pred, t, state_samples).prev_sample
        state_samples[:, 0] = observation
        
    return state_samples


def generate_samples_act(observation, model, scheduler, dataset, config):
    batch_size = config.batch_size
    horizon = config.planning_horizon
    generator = torch.Generator(device=device)
    action_samples = torch.randn((batch_size, horizon, dataset.action_dim), device=device, generator=generator).to(device)
    for i, t in enumerate(scheduler.timesteps):
        timesteps = torch.full((batch_size,), t, device=device, dtype=torch.long)
        scaled_action_samples = scheduler.scale_model_input(action_samples, t)
        with torch.no_grad():
            action_pred = model(observation, scaled_action_samples, timesteps).sample

        if config.ddim:
            action_samples = scheduler.step(action_pred, t, action_samples, eta=config.eta).prev_sample
        else:
            action_samples = scheduler.step(action_pred, t, action_samples).prev_sample
        
    return action_samples


def pipeline(obs, action_model, dyanmics_model, 
             value_model,scheduler_act, scheduler_dyn, 
             dataset, config):
    # we are ready now
    norm_observation = dataset.normalizer.normalize(obs, 'observations')
    observation = torch.tensor(norm_observation, device=device)
    observation = observation.repeat((config.batch_size, 1))

    action_samples = generate_samples_act(observation, action_model, scheduler_act, dataset, config)
    
    state_samples = generate_samples_dyn(observation, action_samples, dyanmics_model, scheduler_dyn, dataset, config)
    
    value_timestep = 0
    timesteps = torch.full((config.batch_size,), value_timestep, device=action_model.device, dtype=torch.long)
    values = value_model(state_samples, action_samples, timesteps)
    sorted_idx = values.argsort(0, descending=True).squeeze()
    sorted_normed_actions = action_samples[sorted_idx]
    best_normed_action = sorted_normed_actions[0].cpu().numpy()
    denorm_action = dataset.normalizer.unnormalize(best_normed_action, 'actions')
    return denorm_action

@dataclass
class TrainingConfig:
    env_name: str = "hopper-medium-v2"
    """Name of the environment"""
    file_name_render: Optional[str] = None
    batch_size: int = 64  # the number of samples to generate, selects the best action
    planning_horizon: int = 32
    max_episode_length: int = 1000
    num_inference_steps: int = 100  # this needs to be <= num_train_timesteps used during training
    render_steps: int = 50
    pretrained_value_model: str = ""
    pretrained_act_model: str = ""
    checkpoint_value_model: str = ""
    checkpoint_act_model: str = ""
    pretrained_dyn_model: str = ""
    checkpoint_dyn_model: str = ""
    use_ema: bool = True
    torch_compile: bool = True
    seed: int = 0
    ddim: bool = False


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

    dataset = ValueDataset(env_name, horizon=config.planning_horizon, normalizer="GaussianNormalizer" , termination_penalty=-100, discount=0.997, seed=config.seed)
    env = dataset.env
    renderer = MuJoCoRenderer(env_name)

    device = "cpu" if not torch.cuda.is_available() else "cuda"

    # Load value function
    print("Loading value model from ", config.pretrained_value_model, config.checkpoint_value_model, "use-ema:", config.use_ema)
    value_model = ValueTransformer.from_pretrained(config.pretrained_value_model, use_safe_tensors=True, 
                                                subfolder="ema" if config.use_ema else "transformer", variant=str(config.checkpoint_value_model)).to(device)

    # Load action model
    ema_str = "_ema" if config.use_ema else ""
    model_act_path = os.path.join(config.pretrained_act_model, f"checkpoints/model_{config.checkpoint_diff_model}{ema_str}.pth")
    action_model = ActionProposalTransformer.from_pretrained(model_act_path).to(device)
    model_dyn_path = os.path.join(config.pretrained_dyn_model, f"checkpoints/model_{config.checkpoint_diff_model}{ema_str}.pth")
    dyanmics_model = DynamicsTransformer.from_pretrained(model_dyn_path).to(device)
    
    if config.ddim:
        scheduler_dyn = DDIMScheduler.from_pretrained(config.pretrained_dyn_model)
        scheduler_act = DDIMScheduler.from_pretrained(config.pretrained_act_model)
    else:
        scheduler_dyn = DDPMScheduler.from_pretrained(config.pretrained_dyn_model)
        scheduler_act = DDPMScheduler.from_pretrained(config.pretrained_act_model)

    scheduler_dyn.set_timesteps(config.num_inference_steps)
    scheduler_act.set_timesteps(config.num_inference_steps)


    if config.torch_compile:
        value_model = torch.compile(value_model)
        action_model = torch.compile(action_model)
        dyanmics_model = torch.compile(dyanmics_model)
        




    obs = env.reset()
    total_reward = 0
    total_score = 0
    rollout = [obs.copy()]

    try:
        for t in tqdm.tqdm(range(config.max_episode_length)):
            # call the policy
            denorm_actions = pipeline(
                obs,
                action_model, dyanmics_model, value_model,
                scheduler_act, scheduler_dyn,
                dataset, config
            )

            # execute action in environment
            next_observation, reward, terminal, _ = env.step(denorm_actions)
            # update return
            total_reward += reward
            # compute score
            score = env.get_normalized_score(total_reward)

            print(
                f"Step: {t}, Reward: {reward}, Total Reward: {total_reward}, Score: {score}"
            )

            # save observations for rendering
            rollout.append(next_observation.copy())

            obs = next_observation

            # if terminal:
            #     show_sample(renderer, [rollout], filename=f"{file_name_render}.mp4", savebase="./renders")
            #     renderer.composite(f"./renders/{file_name_render}.png", np.array(rollout)[None])  
            #     break
            if (t+1) % config.render_steps == 0: 
                show_sample(renderer, [rollout], filename=f"{file_name_render}.mp4", savebase="./renders")
                renderer.composite(f"./renders/{file_name_render}.png", np.array(rollout)[None])

    except KeyboardInterrupt:
        pass

    print(f"Total reward: {total_reward}, Score: {env.get_normalized_score(total_reward)}")