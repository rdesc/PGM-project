import time
import random
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import gym
import torch
import torch.nn as nn
from safetensors.torch import save_file, load
import torch.optim as optim
import torch.nn.functional as F
import tqdm
import mediapy as media
import d4rl
import tyro
import wandb

from diffuser.utils.rendering import MuJoCoRenderer
from diffuser.utils import set_seed
from diffuser.datasets import SequenceDataset


def cycle(dl):
    while True:
        for data in dl:
            yield data

device = "cuda" if torch.cuda.is_available() else "cpu"


class Network(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(Network, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, act_dim),
        ) 

    def forward(self, obs):
        return self.net(obs)


def show_sample(renderer, observations, filename='sample.mp4', savebase='./'):
    '''
    observations : [ batch_size x horizon x observation_dim ]
    '''

    os.makedirs(savebase, exist_ok=True)
    savepath = os.path.join(savebase, filename)

    images = []
    for rollout in observations:
        ## [ horizon x height x width x channels ]
        img = renderer._renders(rollout, partial=True)
        images.append(img)

    ## [ horizon x height x (batch_size * width) x channels ]
    images = np.concatenate(images, axis=2)
    media.write_video(savepath, images, fps=60)

    print('Saved video to', savepath)


@dataclass
class TrainingConfig:
    env_id: str = "hopper-medium-v2"
    seed: int = 0
    train_batch_size: int = 256
    n_train_steps: int = int(200e3)
    model_type: str = "behavior-cloning"
    learning_rate: float = 3e-4
    weight_decay: float = 0
    wandb_track: bool = True
    lr_warmup_steps: int = 100000
    checkpointing_freq: int = 20_000
    horizon: int = 1
    num_workers: int = 1

    # eval params
    run_eval_only: bool = False
    n_episodes: int = 1  # number of episodes to evaluate
    max_episode_length: int = 1000
    render: bool = True
    render_steps: int = 1000
    file_name_render: Optional[str] = None
    pretrained_model: Optional[str] = None
    checkpoint_model: Optional[str] = None


if __name__ == "__main__":
    config = tyro.cli(TrainingConfig)
    set_seed(config.seed); torch.backends.cudnn.deterministic = True

    run_id = int(time.time())
    save_path = f"runs/{config.env_id}/{config.model_type}_{run_id}"
    while os.path.exists(save_path):
        run_id = int(time.time())
        save_path = f"runs/{config.env_id}/{config.model_type}_{run_id}"
    os.makedirs(save_path, exist_ok=True)
    config.output_dir = save_path
    checkpoint_path = f"{save_path}/checkpoints"
    os.makedirs(checkpoint_path)

    dataset = SequenceDataset(config.env_id, horizon=config.horizon, normalizer="GaussianNormalizer", seed=config.seed)
    train_dataloader = cycle (torch.utils.data.DataLoader(
        dataset, batch_size=config.train_batch_size, num_workers=config.num_workers, shuffle=True, pin_memory=True
        ))

    network = Network(dataset.observation_dim, dataset.action_dim).to(device)
    n_trainable = sum(p.numel() for p in network.parameters() if p.requires_grad)
    print('trainable params:', n_trainable)

    ############ Evaluation ############
    if config.run_eval_only:

        # check if file exists
        file_name_render = config.file_name_render if config.file_name_render else os.path.basename(config.pretrained_model) + "_render"
        if os.path.exists(file_name_render + ".mp4") or os.path.exists(file_name_render + ".png"):
            print(f"File {file_name_render} already exists. Exiting.")
            exit()

        if config.render:
            renderer = MuJoCoRenderer(config.env_id)

        model_path = os.path.join(config.pretrained_model, f"checkpoints/model_{config.checkpoint_model}.pth")
        print("Loading model from", model_path)
        with open(model_path, "rb") as f:
            data = f.read()
        network.load_state_dict(load(data))
        
        if config.wandb_track:
            wandb.init(
                config=config,
                name=str(run_id),
                project="diffusion_testing",
                entity="pgm-diffusion"
            )

        env = dataset.env

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

                norm_obs = torch.from_numpy(dataset.normalizer.normalize(obs, 'observations')).float()

                with torch.no_grad():
                    action = network(norm_obs.to(device)).cpu()

                denorm_actions = dataset.normalizer.unnormalize(action, 'actions').numpy()

                # execute action in environment
                next_observation, reward, terminal, _ = env.step(denorm_actions)
                # update return
                total_reward += reward
                # save observations for rendering
                rollout.append(next_observation.copy())
                
                obs = next_observation

                if terminal:
                    if config.render:
                        show_sample(renderer, [rollout], filename=f"{file_name_render}.mp4", savebase="./renders")
                        image = renderer.composite(f"./renders/{file_name_render}.png", np.array(rollout)[None])  
                    break
                if config.render and ((t+1) % config.render_steps == 0 or t == config.max_episode_length - 1): 
                    show_sample(renderer, [rollout], filename=f"{file_name_render}.mp4", savebase="./renders")
                    image = renderer.composite(f"./renders/{file_name_render}.png", np.array(rollout)[None])

            normalized_score = env.get_normalized_score(total_reward)
            ep_returns.append(total_reward)
            ep_scores.append(normalized_score)

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

    ############ Training ############
    else:
        optimizer = optim.Adam(network.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        lr_scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda steps: min((steps+1)/config.lr_warmup_steps, 1)
        )

        if config.wandb_track:
            wandb.init(
                config=config,
                name=str(run_id),
                project="diffusion_training",
                entity="pgm-diffusion"
            )
            wandb.config["optimizer"] = optimizer.__class__.__name__
        
        for i in tqdm.tqdm(range(int(config.n_train_steps))):
            batch = next(train_dataloader)
            trajectories = batch.trajectories.to(device)

            actions = trajectories[:,0, :dataset.action_dim]
            obs = trajectories[:,0, dataset.action_dim:]

            pred_actions = network(obs)

            loss = F.mse_loss(pred_actions, actions)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": i}
            if config.wandb_track:
                wandb.log(logs, step=i)

            if (i+1) % config.checkpointing_freq == 0 :
                model_checkpoint_name = f"model_{i}.pth"
                save_file(network.state_dict(), f"{checkpoint_path}/{model_checkpoint_name}")
                print("saved")

        print("Training Done:", save_path)
