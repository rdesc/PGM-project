import random
import numpy as np
import gym

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset

from tqdm import tqdm
import imageio  
from itertools import islice
from diffuser.utils.rendering import MuJoCoRenderer

import d4rl

def normalize(inp, mean, std):
    return (inp-mean)/std

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



def evaluate(env, actor, render=False):
    cum_reward = 0
    obs = env.reset()
    for t in range(1000):
        norm_obs = normalize(torch.from_numpy(obs).float(), dataset.obs_mean, dataset.obs_std)
        with torch.no_grad():
            action = actor(norm_obs.to(device))
        obs, reward, done, info = env.step(action.cpu().numpy())
        if render:
            frame = env.render("human")
        cum_reward += reward
        if done:
            break
    return cum_reward

if __name__ == "__main__":

    seed = 0
    torch_deterministic = True
    cuda = True

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")

    env_name = "hopper-medium-v2"
    env = gym.make(env_name)
    data_dict = env.get_dataset() # dataset is only used for normalization in this colab
    # eval_env = gym.make("Hopper-v3") # TODO:not sure about this 

    dataset = RLDataset(data_dict, normalize_obs=True, normalize_actions=False)
    loader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True)

    actor = HopperNet(env.observation_space.shape[0], env.action_space.shape[0])
    actor = actor.to(device)
    # actor.load_state_dict(torch.load("actor.pt"))

    # print(evaluate(env, actor, render=True))
    # exit(0)
    
    optimizer = optim.AdamW(actor.parameters(), lr=1e-4, weight_decay=1e-4)
    warmup_steps = 100000 # TODO: check this
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps+1)/warmup_steps, 1)
    )
    # N = len(dataset)
    # k = N // 64
    # print(k)
    # exit(0)
    # 25000
    for epoch in range(10):
        for obs, true_actions in tqdm(loader):
            obs, true_actions = obs.to(device), true_actions.to(device)

            pred_actions = actor(obs)

            optimizer.zero_grad()
            loss = F.mse_loss(pred_actions, true_actions)
            loss.backward()
            optimizer.step()
            scheduler.step()

        print("Epoch", epoch, "loss:", loss.item())
        cum_reward = evaluate(env, actor)
        print("Epoch", epoch, "cum reward:", cum_reward)


    torch.save(actor.state_dict(), "actor.pt")
