
conda activate difuser2
sbatch my_train train_hf.py --env_id=hopper-medium-v2 --horizon=32 --num_workers=8 --num_train_timesteps=20 

## Eval



# this is evaluating our model with their value funciton
python eval_hf_value.py --pretrained_diff_model runs/hopper-medium-expert-v2 --runid_diff_model 1733082766 --checkpoint_diff_model 199999 --file_name_render=render --torch_compile --env_name=hopper-medium-expert-v2 --n_samples=2 --render_step=10000



## eval our model, hopper-medium-v2
(hopper medium, our diffusion model, hugging face model, both trained on medium)
# https://wandb.ai/pgm-diffusion/diffusion_training/runs/nkjlnxxh/overview
python eval_hf_value.py --pretrained_diff_model runs/hopper-medium-v2 --runid_diff_model 1733161086 --checkpoint_diff_model 199999 --file_name_render=render --torch_compile --env_name=hopper-medium-v2 --render_steps=100
Step: 983, Reward: 1.0085915304294293, Total Reward: 1165.190683029831, Score: 0.3639



# reecent our model trained to match their config
python eval_hf_value.py --pretrained_diff_model runs/hopper-medium-v2 --runid_diff_model 1733517481 --checkpoint_diff_model 99999 --file_name_render=render --torch_compile --env_name=hopper-medium-v2 --render_steps=50


# kinda moving??
python eval_hf_value.py --pretrained_diff_model runs/hopper-medium-v2 --runid_diff_model 1733689732 --checkpoint_diff_model 99999 --file_name_render=render --torch_compile --env_name=hopper-medium-v2 --render_steps=50 --scale=0.05
python eval_hf_value.py --pretrained_diff_model runs/hopper-medium-v2 --runid_diff_model 1733689732 --checkpoint_diff_model 99999 --file_name_render=render --torch_compile --env_name=hopper-medium-v2 --render_steps=50 --scale=0.075?

## gets to around 400 without falling!
python eval_hf_value.py --pretrained_diff_model runs/hopper-medium-v2 --runid_diff_model 1733689732 --checkpoint_diff_model 159999 --file_name_render=render --torch_compile --env_name=hopper-medium-v2 --render_steps=50 --scale=0.1 --file_name_render=render_159_0p1

python eval_hf_value.py --pretrained_diff_model runs/hopper-medium-v2 --runid_diff_model 1733689732 --checkpoint_diff_model 199999 --file_name_render=render --torch_compile --env_name=hopper-medium-v2 --render_steps=50 --scale=0.1 --file_name_render=render_199_0p1
(end: Total reward: 1942.957765482953, Score: 0.6032221399619366)


## match?
python3 train_hf.py --train_batch_size=64 --use_original_config --weight_decay=0.0 --horizon=32 --n_train_steps=1000000  --checkpointing_freq=100000 --render_freq=100000  --action_weight=10 --no-cosine_warmup --learning_rate=0.0002 --mixed_precision=no --num_train_timesteps=20

python3 train_hf.py --train_batch_size=2048 --use_original_config --weight_decay=0.0 --horizon=32 --n_train_steps=1000000  --checkpointing_freq=100000 --render_freq=100000  --action_weight=10 --learning_rate=0.0002 