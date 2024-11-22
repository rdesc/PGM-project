# class lax_numpy:
#       def __init__(self):
#             from jax.numpy import isin
#             self.isin = isin
# class numpy:
#       def __init__(self):
#             self.lax_numpy = lax_numpy()

# class _src:
#       def __init__(self):
#             self.numpy = numpy()



import d4rl
import torch
import tqdm
import numpy as np
import gym
import jax
from diffusers import DDPMScheduler, UNet1DModel
from diffusers.experimental import ValueGuidedRLPipeline



def normalize(x_in, data, key):
    means = data[key].mean(axis=0)
    stds = data[key].std(axis=0)
    return (x_in - means) / stds


def de_normalize(x_in, data, key):
    means = data[key].mean(axis=0)
    stds = data[key].std(axis=0)
    return x_in * stds + means

def to_torch(x_in, dtype=None, device=None):
    dtype = dtype or DTYPE
    device = device or DEVICE
    if type(x_in) is dict:
        return {k: to_torch(v, dtype, device) for k, v in x_in.items()}
    elif torch.is_tensor(x_in):
        return x_in.to(device).type(dtype)
    return torch.tensor(x_in, dtype=dtype, device=device)

def reset_x0(x_in, cond, act_dim):
    for key, val in cond.items():
        x_in[:, key, act_dim:] = val.clone()
    return x_in


if __name__ == "__main__":

    # create environment
    env_name = "hopper-medium-v2"
    env = gym.make(env_name)
    data = env.get_dataset() # dataset is only used for normalization in this colab

    # Define configs and constants
    # Cuda settings for colab
    torch.cuda.get_device_name(0)
    DEVICE = 'cuda:0'
    DTYPE = torch.float

    # diffusion model settings
    n_samples = 4   # number of trajectories planned via diffusion
    horizon = 128   # length of sampled trajectories
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    num_inference_steps = 20 # number of difusion steps

    # sample intial env step
    obs = env.reset()
    obs_raw = obs

    # normalize observations for forward passes
    obs = normalize(obs, data, 'observations')

    

    # Two generators for different parts of the diffusion loop to work in colab
    generator = torch.Generator(device='cuda')
    generator_cpu = torch.Generator(device='cpu')

    scheduler = DDPMScheduler(num_train_timesteps=100,beta_schedule="squaredcos_cap_v2")

    # The horizion represents the length of trajectories used in training.
    network = UNet1DModel.from_pretrained("bglick13/hopper-medium-v2-value-function-hor32", subfolder="unet").to(device=DEVICE)
    # print(network)
    # trajetory = torch.randn(1, 4, 14).to(DEVICE)
    # sample random initial noise vector
    # output = network(trajetory, timestep=0)
    # import pdb; pdb.set_trace()
    # setup for denoising 
    # add a batch dimension and repeat for multiple samples
    # [ observation_dim ] --> [ n_samples x observation_dim ]
    obs = obs[None].repeat(n_samples, axis=0)
    conditions = {
        0: to_torch(obs, device=DEVICE)
    }
    # constants for inference
    batch_size = len(conditions[0])
    shape = (batch_size, horizon, state_dim+action_dim)
    
    # sample random initial noise vector
    x1 = torch.randn(shape, device=DEVICE, generator=generator)

    # this model is conditioned from an initial state, so you will see this function
   #   multiple times to change the initial state of generated data to the state
     #generated via env.reset() above or env.step() below
    x = reset_x0(x1, conditions, action_dim)
    # import pdb;pdb.set_trace()

    # convert a np observation to torch for model forward pass
    x = to_torch(x)
    import pdb;pdb.set_trace()
    
    eta = 1.0 # noise factor for sampling reconstructed state

    # run the diffusion process
    # for i in tqdm.tqdm(reversed(range(num_inference_steps)), total=num_inference_steps):
    for i in tqdm.tqdm(scheduler.timesteps):

        # create batch of timesteps to pass into model
        timesteps = torch.full((batch_size,), i, device=DEVICE, dtype=torch.long)

        # 1. generate prediction from model
        with torch.no_grad():
            residual = network(x.permute(0, 2, 1), timesteps).sample
            residual = residual.permute(0, 2, 1) # needed to match model params to original

        # 2. use the model prediction to reconstruct an observation (de-noise)
        obs_reconstruct = scheduler.step(residual, i, x)["prev_sample"]

        # 3. [optional] add posterior noise to the sample
        if eta > 0:
            noise = torch.randn(obs_reconstruct.shape, generator=generator_cpu).to(obs_reconstruct.device)
            posterior_variance = scheduler._get_variance(i) # * noise
            # no noise when t == 0
            # NOTE: original implementation missing sqrt on posterior_variance
            obs_reconstruct = obs_reconstruct + int(i>0) * (0.5 * posterior_variance) * eta* noise  # MJ had as log var, exponentiated

        # 4. apply conditions to the trajectory
        obs_reconstruct_postcond = reset_x0(obs_reconstruct, conditions, action_dim)
        x = to_torch(obs_reconstruct_postcond)
    print(x.shape)

    pipeline = ValueGuidedRLPipeline.from_pretrained(
        "bglick13/hopper-medium-v2-value-function-hor32",
        env=env,
    )
    
    import pdb; pdb.set_trace()

    env.seed(0)
    obs = env.reset()
    total_reward = 0
    total_score = 0
    T = 100
    rollout = [obs.copy()]
    trajectories = []
    y_maxes = [0]
    for t in tqdm.tqdm(range(T)):
        # normalize observations for forward passes
        denorm_actions = pipeline(obs, planning_horizon=32)

        # execute action in environment
        next_observation, reward, terminal, _ = env.step(denorm_actions)
        score = env.get_normalized_score(total_reward)

        # update return
        total_reward += reward
        total_score += score
        print(
            f"Step: {t}, Reward: {reward}, Total Reward: {total_reward}, Score: {score}, Total Score:"
            f" {total_score}"
        )
        # save observations for rendering
        rollout.append(next_observation.copy())

        obs = next_observation
















