import os 
import time
from dataclasses import dataclass, asdict
from typing import Optional

# import d4rl  # noqa

import numpy as np
import torch

import gym
import tqdm
from diffusers import  UNet1DModel, DDPMScheduler, DDPMPipeline
from diffuser.datasets import ValueDataset
from diffuser.utils.rendering import MuJoCoRenderer
from diffuser.utils import set_seed
from bc_d4rl import show_sample
import wandb
import tyro


def generate_samples(initial_state, actions, model, dataset, scheduler, batch_size, horizon):
    generator = torch.Generator(device=device)
    shape = (batch_size, horizon, dataset.observation_dim + dataset.action_dim,)
    x = torch.randn(shape, device=device, generator=generator).to(device)

    initial_state = initial_state.repeat(batch_size, 1)

    for i, t in enumerate(scheduler.timesteps):
        timesteps = torch.full((batch_size,), t, device=device, dtype=torch.long)
        model_input = scheduler.scale_model_input(x, t)
        
        # conditioning
        model_input[:, 0, dataset.action_dim: ] = initial_state
        model_input[:, :, :dataset.action_dim] = actions

        with torch.no_grad():
            noise_pred = model(model_input.permute(0, 2, 1), timesteps).sample
            noise_pred = noise_pred.permute(0, 2, 1) # needed to match model params to original

        x = scheduler.step(noise_pred, t, x).prev_sample
    
    x[:, 0, dataset.action_dim: ] = initial_state
    x[:, :, :dataset.action_dim] = actions
    
    return x


def generate_actions(initial_state, model, dataset, scheduler, batch_size, horizon):
    generator = torch.Generator(device=device)
    shape = (batch_size, horizon, dataset.observation_dim + dataset.action_dim,)
    x = torch.randn(shape, device=device, generator=generator).to(device)

    initial_state = initial_state.repeat(horizon, batch_size, 1).permute(1, 0, 2)
    
    for i, t in enumerate(scheduler.timesteps):
        timesteps = torch.full((batch_size,), t, device=device, dtype=torch.long)
        model_input = scheduler.scale_model_input(x, t)

        # conditioning
        model_input[:, :, dataset.action_dim:] = initial_state

        with torch.no_grad():
            noise_pred = model(model_input.permute(0, 2, 1), timesteps).sample
            noise_pred = noise_pred.permute(0, 2, 1) # needed to match model params to original

        x = scheduler.step(noise_pred, t, x).prev_sample

    return x[:, :, :dataset.action_dim]


@dataclass
class TrainingConfig:
    env_name: str = "hopper-medium-v2"
    """Name of the environment"""
    file_name_render: Optional[str] = None
    batch_size: int = 64  # for sample-score-rank -- the number of samples to generate, selects the best action
    horizon: int = 32  
    max_episode_length: int = 1000
    # this needs to be <= num_train_timesteps used during training, D-MPC uses 32 diffusion steps for action model and 10 diffusion steps for dynamics model
    # can maybe simplify eval by using the same number of steps for both models (for now)
    num_inference_steps: int = 32
    render_steps: int = 50
    history: int = 0  # disabled for now, set to 0
    
    pretrained_value_model: str = ''
    pretrained_action_model: str = ''
    pretrained_dynamics_model: str = ''

    checkpoint_value_model: str = ''
    checkpoint_action_model: str = ''
    checkpoint_dynamics_model: str = ''

    use_ema: bool = True
    torch_compile: bool = True
    seed: int = 0
    wandb_track: bool = True


if __name__ == "__main__":
    config = tyro.cli(TrainingConfig)

    set_seed(config.seed)

    run_id = int(time.time())
    if config.wandb_track:
        wandb.init(
            config=config,
            name=f"eval_dmpc_{run_id}",
            project="diffusion_training",
            entity="pgm-diffusion"
        )

    # check if file exists
    file_name_render = config.file_name_render if config.file_name_render else os.path.basename(config.pretrained_value_model) + "_render"
    if os.path.exists(file_name_render + ".mp4") or os.path.exists(file_name_render + ".png"):
        print(f"File {file_name_render} already exists. Exiting.")
        exit()

    env_name = config.env_name
    dataset = ValueDataset(env_name, horizon=config.horizon, normalizer="GaussianNormalizer" , termination_penalty=-100, discount=0.997, seed=config.seed)  # TODO: add config param for discount
    env = gym.make(env_name)
    env.seed(config.seed)
    renderer = MuJoCoRenderer(env_name)

    device = "cpu" if not torch.cuda.is_available() else "cuda"

    print("\nLoading value model from", config.pretrained_value_model, config.checkpoint_value_model, "use-ema:", config.use_ema)
    value_function = UNet1DModel.from_pretrained(config.pretrained_value_model, use_safe_tensors=True,
                                                 subfolder="ema" if config.use_ema else "unet", variant=str(config.checkpoint_value_model)).to(device)

    print("\nLoading action diffusion model from", config.pretrained_action_model, config.checkpoint_action_model)
    action_model = UNet1DModel.from_pretrained(os.path.join(config.pretrained_action_model, "checkpoints/model_{}.pth".format(config.checkpoint_action_model))).to(device)

    print("\nLoading dynamics diffusion model from", config.pretrained_dynamics_model, config.checkpoint_dynamics_model)
    dynamics_model = UNet1DModel.from_pretrained(os.path.join(config.pretrained_dynamics_model, "checkpoints/model_{}.pth".format(config.checkpoint_dynamics_model))).to(device)
    
    # NOTE: use same scheduler for both models
    scheduler = DDPMScheduler.from_pretrained(config.pretrained_dynamics_model,
                                              # below are kwargs to overwrite the config loaded from the pretrained model
                                              )
    scheduler.set_timesteps(config.num_inference_steps)
    
    if config.torch_compile:
        value_function = torch.compile(value_function)
        action_model = torch.compile(action_model)
        dynamics_model = torch.compile(dynamics_model)
        
    obs = env.reset()
    env.set_state(env.init_qpos + np.random.randn(env.model.nq) * 0.1, env.init_qvel + np.random.randn(env.model.nv) * 0.1)
    total_reward = 0
    total_score = 0
    rollout = [obs.copy()]

    with torch.no_grad():
        try:
            for t in tqdm.tqdm(range(config.max_episode_length)):
                # normalize observation
                norm_observation = dataset.normalizer.normalize(obs, 'observations')

                initial_state = torch.from_numpy(norm_observation).to(device)
                
                actions = generate_actions(initial_state, action_model, dataset, scheduler, config.batch_size, config.horizon)
                samples = generate_samples(initial_state, actions, dynamics_model, dataset, scheduler, config.batch_size, config.horizon)

                # sample score and rank
                # get values of samples
                timesteps = torch.full((config.batch_size,), 0, device=device, dtype=torch.long)
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

    print(f"Total reward: {total_reward}, Score: {env.get_normalized_score(total_reward)}")

    # save to disk txt file
    with open(f"{file_name_render}.txt", "w") as f:
        f.write(f"Total reward: {total_reward}, Score: {env.get_normalized_score(total_reward)} \n")


# TODO: better way of saving results
# TODO: better directory structure for saving results
# TODO: can we store the results to wandb?