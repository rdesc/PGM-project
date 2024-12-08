import random
import os
import numpy as np
import gym

import torch
import torch.nn as nn
from safetensors.torch import save_file, load
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset

from tqdm import tqdm
import mediapy as media
from diffuser.utils.rendering import MuJoCoRenderer

import d4rl

def normalize(inp, mean, std):
    return (inp - mean) / std

def de_normalize(inp, mean, std):
    return inp * std + mean

class RLDataset(Dataset):
    def __init__(self, data_dict, normalize_obs=False, normalize_actions=False):
        self.data_dict = data_dict
        actions = torch.from_numpy(data_dict["actions"]).float()
        obs = torch.from_numpy(data_dict["observations"]).float()

        obs_mean = obs.mean(axis=0)
        obs_std = obs.std(axis=0)

        actions_mean = actions.mean(axis=0)
        actions_std = actions.std(axis=0)

        if normalize_obs:
            obs = normalize(obs, obs_mean, obs_std)

        if normalize_actions:
            actions = normalize(actions, actions_mean, actions_std)

        self.obs_mean = obs_mean
        self.obs_std = obs_std
        self.actions_mean = actions_mean
        self.actions_std = actions_std

        self.obs = obs
        self.actions = actions

        self.device = device

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return self.obs[idx], self.actions[idx]


class HopperNet(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(HopperNet, self).__init__()

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
            nn.Tanh()
        ) 

    def forward(self, obs):
        return self.net(obs)
    

def mkdir(savepath):
    """
        returns `True` iff `savepath` is created
    """
    if not os.path.exists(savepath):
        os.makedirs(savepath)
        return True
    else:
        return False
    

def show_sample(renderer, observations, filename='sample.mp4', savebase='./'):
    '''
    observations : [ batch_size x horizon x observation_dim ]
    '''

    mkdir(savebase)
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


def evaluate(env, actor, T=1000, render=False, filename='bc_sample.mp4', de_normalize_actions=False):
    cum_reward = 0
    obs = env.reset()
    rollout = [obs.copy()]

    for t in range(T):
        norm_obs = normalize(torch.from_numpy(obs).float(), dataset.obs_mean, dataset.obs_std)
        with torch.no_grad():
            action = actor(norm_obs.to(device)).cpu()
        if de_normalize_actions:
            action = de_normalize(action, dataset.actions_mean, dataset.actions_std)
        obs, reward, done, info = env.step(action.numpy())
        
        rollout.append(obs.copy())
        
        cum_reward += reward
        # if done:
            # break
    
    if render:
        show_sample(render, [rollout], filename=filename)

    return cum_reward


if __name__ == "__main__":
    env_name = "hopper-medium-v2"
    num_epochs = 20
    num_inference_steps = 1000
    run_eval_only = True
    eval_iters = 50
    render = False
    pretrained_model_path = "bc_hopper-medium-v2_epoch13.safetensors"
    normalize_state_actions = True

    seed = 0
    torch_deterministic = True
    cuda = True

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")

    env = gym.make(env_name)
    data_dict = env.get_dataset() # dataset is only used for normalization in this colab
    # eval_env = gym.make("Hopper-v3") # TODO:not sure about this 

    dataset = RLDataset(data_dict,
                        normalize_obs=normalize_state_actions,
                        normalize_actions=normalize_state_actions)
    loader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True)

    actor = HopperNet(env.observation_space.shape[0], env.action_space.shape[0])
    actor = actor.to(device)

    if run_eval_only:

        scores = []
        for i in range(eval_iters):
            with open(pretrained_model_path, "rb") as f:
                data = f.read()
            actor.load_state_dict(load(data))

            cum_reward = evaluate(env, actor,
                                T=num_inference_steps,
                                render=MuJoCoRenderer(env) if render else False,
                                filename=str(i) + "-" + os.path.basename(pretrained_model_path).replace(".safetensors", ".mp4"))
            score = env.get_normalized_score(cum_reward)

            scores.append(score)

            print(
                f"Total Reward: {cum_reward}, Score: {score}"
            )

        print("Average Score: ", np.mean(scores), "Std: ", np.std(scores))

    else:    
        optimizer = optim.AdamW(actor.parameters(), lr=1e-4, weight_decay=1e-4)
        warmup_steps = 100000 # TODO: check this
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda steps: min((steps+1)/warmup_steps, 1)
        )

        step = 0
        for epoch in range(num_epochs):
            for obs, true_actions in tqdm(loader):
                obs, true_actions = obs.to(device), true_actions.to(device)

                pred_actions = actor(obs)

                optimizer.zero_grad()
                loss = F.mse_loss(pred_actions, true_actions)
                loss.backward()
                optimizer.step()
                scheduler.step()

                step+=1

            cum_reward = evaluate(env, actor, T=num_inference_steps, de_normalize_actions=normalize_state_actions)
            score = env.get_normalized_score(cum_reward)

            print(
                f"Epoch: {epoch}, Step: {step}, Loss: {loss.item()}, Total Reward: {cum_reward}, Score: {score}"
            )

            save_file(actor.state_dict(), f"bc_{env_name}_epoch{epoch}.safetensors")
