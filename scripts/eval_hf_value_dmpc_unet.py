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
from scripts.train_bc import show_sample
import wandb
import tyro

@torch.no_grad()
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

@torch.no_grad()
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
    env_id: str = "hopper-medium-v2"
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
    pretrained_act_model: str = ''
    pretrained_dyn_model: str = ''

    checkpoint_value_model: str = ''
    checkpoint_act_model: str = ''
    checkpoint_dyn_model: str = ''

    use_ema: bool = True
    torch_compile: bool = True
    seed: int = 0
    wandb_track: bool = True
    n_episodes: int = 1


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

    env_id = config.env_id
    dataset = ValueDataset(env_id, horizon=config.horizon, normalizer="GaussianNormalizer" , termination_penalty=-100, discount=0.997, seed=config.seed)  # TODO: add config param for discount
    env = gym.make(env_id)
    env.seed(config.seed)
    renderer = MuJoCoRenderer(env_id)

    device = "cpu" if not torch.cuda.is_available() else "cuda"

    print("\nLoading value model from", config.pretrained_value_model, config.checkpoint_value_model, "use-ema:", config.use_ema)
    value_model = UNet1DModel.from_pretrained(config.pretrained_value_model, use_safe_tensors=True,
                                                 subfolder="ema" if config.use_ema else "unet", variant=str(config.checkpoint_value_model)).to(device)

    print("\nLoading action diffusion model from", config.pretrained_act_model, config.checkpoint_act_model)
    action_model = UNet1DModel.from_pretrained(os.path.join(config.pretrained_act_model, "checkpoints/model_{}.pth".format(config.checkpoint_act_model))).to(device)

    print("\nLoading dynamics diffusion model from", config.pretrained_dyn_model, config.checkpoint_dyn_model)
    dynamics_model = UNet1DModel.from_pretrained(os.path.join(config.pretrained_dyn_model, "checkpoints/model_{}.pth".format(config.checkpoint_dyn_model))).to(device)
    
    scheduler_dyn = DDPMScheduler.from_pretrained(config.pretrained_dyn_model)
    scheduler_act = DDPMScheduler.from_pretrained(config.pretrained_act_model)

    scheduler_dyn.set_timesteps(config.num_inference_steps)
    scheduler_act.set_timesteps(config.num_inference_steps)

    if config.torch_compile:
        value_model = torch.compile(value_model)
        action_model = torch.compile(action_model)
        dynamics_model = torch.compile(dynamics_model)

    config.model_type = 'dmpc'
    config.arch_type = 'unet'

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
            # normalize observation
            norm_observation = dataset.normalizer.normalize(obs, 'observations')

            initial_state = torch.from_numpy(norm_observation).to(device)
            
            actions = generate_actions(initial_state, action_model, dataset, scheduler_act, config.batch_size, config.horizon)
            samples = generate_samples(initial_state, actions, dynamics_model, dataset, scheduler_dyn, config.batch_size, config.horizon)

            # sample score and rank
            # get values of samples
            timesteps = torch.full((config.batch_size,), 0, device=device, dtype=torch.long)
            values = value_model(samples.permute(0, 2, 1), timesteps).sample
            sorted_idx = values.argsort(0, descending=True).squeeze()
            sorted_samples = samples[sorted_idx]
            # extract actions
            normed_actions = sorted_samples[:, 0, :dataset.action_dim ]
            best_action = normed_actions[0].cpu().numpy()
            denorm_actions = dataset.normalizer.unnormalize(best_action, 'actions')

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

        if image is None and config.render:
            image = renderer.composite(f"./renders/{file_name_render}.png", np.array(rollout)[None])

        print(f"Total reward: {total_reward}, Score: {env.get_normalized_score(total_reward)}")
        if config.wandb_track:
            logs = {"score": normalized_score, "total_reward":total_reward, 'seed': seed}
            if image is not None:
                logs['image'] = wandb.Image(image, caption=f"composite {seed}", file_type="png")
            wandb.log(logs)
    if config.wandb_track:
        wandb.summary["avg_return"] = np.mean(ep_returns)
        wandb.summary["avg_return"] = np.std(ep_returns)
        wandb.summary["avg_score"] = np.mean(ep_scores)
        wandb.summary["std_score"] = np.std(ep_scores) 
