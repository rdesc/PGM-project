from dataclasses import dataclass, asdict, fields
from diffuser.datasets import SequenceDataset
from diffusers import  UNet1DModel, DDPMScheduler
from tqdm import tqdm
from accelerate import Accelerator
from huggingface_hub import HfFolder, Repository, whoami
from tqdm.auto import tqdm
from pathlib import Path
import os
import torch
import torch.nn.functional as F
from diffusers import DDPMPipeline
import accelerate
from diffuser.utils.rendering import MuJoCoRenderer
from diffuser.utils import set_seed
import time
import yaml
import numpy as np
from train_hf import TrainingConfig, generate_samples
import json
import tyro

device = "cuda" if torch.cuda.is_available() else "cpu"

class DotDict(dict):
    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, attr, value):
        self[attr] = value

    def __delattr__(self, attr):
        if attr in self:
            del self[attr]

def load_training_config(config_path):
    with open(config_path, "r") as file:
        data = yaml.safe_load(file)
    return  TrainingConfig(**data)

def create_training_paths(env_id, run_id, checkpoint_id=None):
    run_path = f"runs/{env_id}/{run_id}"
    config_path = f"{run_path}/config.yaml"
    checkpoint_dir = f"{run_path}/checkpoints"
    if checkpoint_id:
        checkpoint_path = f"{run_path}/checkpoints/model_{checkpoint_id}.pth"
    else:
        max_ = None
        for path in os.listdir(checkpoint_dir):
            splitted = path.split("_")[-1]
            end_index = splitted.index(".")
            rund_id = splitted[: end_index]
            if max_ is None:
                max_ = rund_id
            else:
                max_ = max(max_, rund_id)
            
        checkpoint_path = f"{run_path}/checkpoints/model_{max_}.pth"
    return config_path, checkpoint_path


def load_json(path):
    # Load data from a JSON file
    with open(path, "r") as file:
        data = DotDict(json.load(file))
    return data

@dataclass
class EvalConfig:
    env_id: str = "hopper-medium-expert-v2"
    eval_batch_size: int = 1  # how many images to sample during evaluation
    num_train_timesteps: int = 100
    horizon: int = 128
    seed: int = 0
    use_sample_hf_for_eval: bool= False
    use_conditioning_for_sampling: bool = True
    render: bool = True
    render_steps: int = 50
    render_file_name: str = "faisal-eval-unguided"
    run_id: int = "1733012971"
    checkpoint_id: int = 199999
    max_episode_length: int = 1000
    n_episodes: int = 1
    

if __name__ == "__main__":

    eval_config = tyro.cli(EvalConfig)

    set_seed(eval_config.seed)

    render_file_name = eval_config.render_file_name if eval_config.render_file_name else eval_config.run_id + "_render"
    if os.path.exists(render_file_name + ".mp4") or os.path.exists(render_file_name + ".png"):
        print(f"File {render_file_name} already exists. Exiting.")
        exit()

    config_path, checkpoint_path = create_training_paths(eval_config.env_id, eval_config.run_id, checkpoint_id=eval_config.checkpoint_id)
    
    training_config = load_training_config(config_path=config_path)
    model_config = load_json(f'{checkpoint_path}/config.json')
    scheduler_config = load_json(f'{checkpoint_path}/../../scheduler_config.json')
    
    for field in fields(eval_config):
        field_name = field.name
        field_value = getattr(eval_config, field_name)
        if field_value is None and field_name != 'checkpoint_id':
            setattr(eval_config, field_name, getattr(training_config, field_name))

    dataset = SequenceDataset(eval_config.env_id, horizon=eval_config.horizon, normalizer="GaussianNormalizer", seed=eval_config.seed)

    model_class = eval(model_config._class_name)
    model = model_class.from_pretrained(checkpoint_path).to(device)
    # scheduler = DDPMScheduler(num_train_timesteps=config.num_train_timesteps,beta_schedule="squaredcos_cap_v2")
    scheduler_class = eval(scheduler_config._class_name)
    noise_scheduler = scheduler_class.from_pretrained(f'{checkpoint_path}/../../')

    env = dataset.env

    renderer = MuJoCoRenderer(eval_config.env_id)
    
    total_rewards = []
    total_scores = []

    for i in range(eval_config.n_episodes):
        observation = env.reset()

        ## observations for rendering
        rollout = [observation.copy()]

        total_reward = 0
        score = 0
        for t in range(eval_config.max_episode_length):

            ## save state for rendering only
            state = env.state_vector().copy()

            ## format current observation for conditioning
            conditions = torch.tensor(observation, device=device)
            conditions = conditions.repeat((eval_config.eval_batch_size, 1))

            samples = generate_samples(eval_config, conditions ,model, dataset, noise_scheduler, use_pipeline=eval_config.use_conditioning_for_sampling)
            # extract actions
            actions = samples[:, 0, :dataset.action_dim]
            # we do not have a way rank, assume first trajectory is optimal
            action = actions[0, :].cpu().numpy()

            ## execute action in environment
            next_observation, reward, terminal, _ = env.step(action)

            ## print reward and score
            total_reward += reward
            score = env.get_normalized_score(total_reward)
            print(
                f't: {t} | r: {reward:.2f} |  R: {total_reward:.2f} | score: {score:.4f} | ',
                flush=True,
            )

            ## update rollout observations
            rollout.append(next_observation.copy())

            ## render every `args.vis_freq` steps
            # logger.log(t, samples, state, rollout)

            if t % eval_config.render_steps == 0: 
                renderer.render_rollout(f"./{render_file_name}_{t}.mp4", np.array(rollout))
                renderer.composite(f"./{render_file_name}_{t}.png", np.array(rollout)[None])

            if terminal:
                print('Hit terminal state!')
                renderer.render_rollout(f"./{render_file_name}_{t}.mp4", np.array(rollout))
                renderer.composite(f"./{render_file_name}_{t}.png", np.array(rollout)[None])
                break

            observation = next_observation
        total_rewards.append(total_reward)
        total_scores.append(score)

    print(f'mean reward over {eval_config.n_episodes} episodes', np.mean(total_rewards))
    print(f'std reward over {eval_config.n_episodes} episodes', np.std(total_rewards))
    print(f'mean score over {eval_config.n_episodes} episodes', np.mean(total_scores))
    print(f'std score over {eval_config.n_episodes} episodes', np.std(total_scores))
    