import d4rl  # noqa

import numpy as np
import gym
import tqdm
from diffusers.experimental import ValueGuidedRLPipeline
from diffusers import DDPMPipeline
from diffuser.utils.rendering import MuJoCoRenderer


config = {
    "n_samples": 64,
    "horizon": 32,
    "num_inference_steps": 20,
    "n_guide_steps": 2,  # can set to 0 for faster sampling, does not use value network
    "scale_grad_by_std": True,
    "scale": 0.1,
    "eta": 0.0,
    "t_grad_cutoff": 2,
    "device": "cpu",
}


if __name__ == "__main__":
    env_name = "hopper-medium-v2"
    env = gym.make(env_name)
    renderer = MuJoCoRenderer(env_name)


    value_model = DDPMPipeline.from_pretrained("/home/ozgur/git/PGM-project/diffuser-hopperv2-32-trial-03", use_safe_tensors=True, variant=str(190000) )
    pipeline = ValueGuidedRLPipeline.from_pretrained(
        "bglick13/hopper-medium-v2-value-function-hor32",
        env=env,
    )
    pipeline.value_function = value_model.unet
    pipeline.register_modules(value_functions=pipeline.value_function)

    env.seed(0)
    obs = env.reset()
    total_reward = 0
    total_score = 0
    T = 1000
    rollout = [obs.copy()]
    try:
        for t in tqdm.tqdm(range(T)):
            # call the policy
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
        renderer.render_rollout("./render_env.mp4", np.array(rollout))

    except KeyboardInterrupt:
        pass

    print(f"Total reward: {total_reward}")