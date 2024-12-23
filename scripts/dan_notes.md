
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

## this gets further
python eval_hf_value.py --pretrained_diff_model runs/hopper-medium-v2 --runid_diff_model 1733689732 --checkpoint_diff_model 199999 --file_name_render=render --torch_compile --env_name=hopper-medium-v2 --render_steps=50 --scale=0.1 --file_name_render=render_199_0p1
(end: Total reward: 1942.957765482953, Score: 0.6032221399619366)


## match?
python3 train_hf.py --train_batch_size=64 --use_original_config --weight_decay=0.0 --horizon=32 --n_train_steps=1000000  --checkpointing_freq=100000 --render_freq=100000  --action_weight=10 --no-cosine_warmup --learning_rate=0.0002 --mixed_precision=no --num_train_timesteps=20

python3 train_hf.py --train_batch_size=2048 --use_original_config --weight_decay=0.0 --horizon=32 --n_train_steps=1000000  --checkpointing_freq=100000 --render_freq=100000  --action_weight=10 --learning_rate=0.0002 

## eval 1 mil models
### this should be really close to the janner codebase now:?
python eval_hf_value.py --pretrained_diff_model runs/hopper-medium-v2 --runid_diff_model 1733712655 --checkpoint_diff_model 799999_ema --file_name_render=render --torch_compile --env_name=hopper-medium-v2 --render_steps=50 --scale=0.1 --file_name_render=render_100d_800k_js_newsampling
Total reward: 1585.5527152626878, Score: 0.4934058517598944

python eval_hf_value.py --pretrained_diff_model runs/hopper-medium-v2 --runid_diff_model 1733712655 --checkpoint_diff_model 799999 --file_name_render=render --torch_compile --env_name=hopper-medium-v2 --render_steps=50 --scale=0.1 --file_name_render=render_100d_noema800k_js
Total reward: 1558.021066076837, Score: 0.484946476270355

## close to janner but with 20 timesteps
python eval_hf_value.py --pretrained_diff_model runs/hopper-medium-v2 --runid_diff_model 1733712114 --checkpoint_diff_model 799999_ema --file_name_render=render --torch_compile --env_name=hopper-medium-v2 --render_steps=50 --scale=0.1 --file_name_render=render_100d_800k_js_20 --num_inference_steps=20


## slightly better guy (big batch, cos schedule, )
python eval_hf_value.py --pretrained_diff_model runs/hopper-medium-v2 --runid_diff_model 1733712280 --checkpoint_diff_model 799999_ema --file_name_render=render --torch_compile --env_name=hopper-medium-v2 --render_steps=50 --scale=0.1 --file_name_render=render_100d_800k_os

## 
## our diffusion model in their codebase is working!
python scripts/plan_guided_diff.py --diffusion_epoch 800000 --value_epoch 160000 --loadbase logs/pretrained --seed 0 --scale 0.0 --dataset hopper-medium-v2

seed 0:
t: 999 | r: 1.03 |  R: 2480.86 | score: 0.7685 
seed 1:
t: 999 | r: 1.51 |  R: 3109.47 | score: 0.9616


## trying debug
python eval_hf_value_debug.py --pretrained_diff_model runs/hopper-medium-v2 --runid_diff_model 1733712655 --checkpoint_diff_model 799999_ema --pretrained_value_model runs/hopper-medium-v2/value_1733876588 --checkpoint_value_model=180000 --env_name=hopper-medium-v2 --render_steps=50 --scale=0.0 --file_name_render=render_debug --num_inference_steps=100 --no-torch_compile


## working
python eval_hf_value_debug.py --pretrained_diff_model runs/hopper-medium-v2 --runid_diff_model 1733712114 --checkpoint_diff_model 799999_ema --pretrained_value_model runs/hopper-medium-v2/value_1733876588 --checkpoint_value_model=180000 --env_name=hopper-medium-v2 --render_steps=2000 --scale=0.0 --file_name_render=render_debug --num_inference_steps=20 --torch_compile --seed=1

## try 100 timestpe 
python eval_hf_value_debug.py --pretrained_diff_model runs/hopper-medium-v2 --runid_diff_model 1733712655 --checkpoint_diff_model 799999_ema --pretrained_value_model runs/hopper-medium-v2/value_1733876588 --checkpoint_value_model=180000 --env_name=hopper-medium-v2 --render_steps=2000 --scale=0.0 --file_name_render=render_debug --num_inference_steps=100 --torch_compile --seed=1


python3 train_hf_transformer.py --train_batch_size=64 --pred_noise --weight_decay=0.0 --horizon=32 --n_train_steps=1000000  --checkpointing_freq=100000 --render_freq=100000  --action_weight=10 --cosine_warmup --learning_rate=0.0002 --mixed_precision=fp16 --num_train_timesteps=20 --torch_compile

--no_wandb_track 

## train our tansformer, 
add here


## eval our transformer
python scripts/eval_hf_value.py --use-ema --scale 0.1 --num_inference_steps 20 --file_name_render test_transformer_s01 --seed 0 --pretrained_diff_model runs/1734063759 --checkpoint_diff_model 799999

## train value function
sbatch my_train scripts/train_hf_value.py --arch_type transformer
sbatch my_train scripts/train_hf_value.py --arch_type transformer --env_id halfcheetah-medium-v2
sbatch my_train scripts/train_hf_value.py --arch_type transformer --env_id walker2d-medium-v2

## train action proposal
train_hf_action_transformer.py  --train_batch_size=64 --pred_noise --weight_decay=0.0 --horizon=32 --n_train_steps=1000000 --checkpointing_freq=100000 --render_freq=100000 --action_weight=10 --cosine_warmup --learning_rate=0.0002 --mixed_precision=fp16 --num_train_timesteps=20 --torch_compile 

scripts/train_hf_action_transformer.py --train_batch_size=64 --pred_noise --weight_decay=0.0 --horizon=32 --n_train_steps=1000000 --checkpointing_freq=100000 --render_freq=100000 --action_weight=10 --cosine_warmup --learning_rate=0.0002 --mixed_precision=fp16 --num_train_timesteps=20 --torch_compile --env_id halfcheetah-medium-v2
scripts/train_hf_action_transformer.py --train_batch_size=64 --pred_noise --weight_decay=0.0 --horizon=32 --n_train_steps=1000000 --checkpointing_freq=100000 --render_freq=100000 --action_weight=10 --cosine_warmup --learning_rate=0.0002 --mixed_precision=fp16 --num_train_timesteps=20 --torch_compile --env_id walker2d-medium-v2


## train state dyn
train_hf_dynamics_transformer.py --train_batch_size=64 --pred_noise --weight_decay=0.0 --horizon=32 --n_train_steps=1000000 --checkpointing_freq=100000 --render_freq=100000  --cosine_warmup --learning_rate=0.0002 --mixed_precision=fp16 --num_train_timesteps=20 --torch_compile
train_hf_dynamics_transformer.py --train_batch_size=64 --pred_noise --weight_decay=0.0 --horizon=32 --n_train_steps=1000000 --checkpointing_freq=100000 --render_freq=100000  --cosine_warmup --learning_rate=0.0002 --mixed_precision=fp16 --num_train_timesteps=20 --torch_compile --env_id halfcheetah-medium-v2

train_hf_dynamics_transformer.py --train_batch_size=64 --pred_noise --weight_decay=0.0 --horizon=32 --n_train_steps=1000000 --checkpointing_freq=100000 --render_freq=100000  --cosine_warmup --learning_rate=0.0002 --mixed_precision=fp16 --num_train_timesteps=20 --torch_compile --env_id walker2d-medium-v2



## eval combo
hopper
python scripts/eval_hf_value_dmpc.py --use-ema --num_inference_steps 20 --file_name_render test_transformer_combos0 --seed 0 --pretrained_value_model runs/hopper-medium-v2/value_1734128084 --checkpoint_value_model=180000 --pretrained_act_model runs/hopper-medium-v2/1734141113 --checkpoint_act_model=799999 --pretrained_dyn_model runs/hopper-medium-v2/1734140811 --checkpoint_dyn_model=799999 --n_episodes=5 --render_steps=2000

walker2d
hopper
python scripts/eval_hf_value_dmpc.py --env_name walker2d-medium-v2  --use-ema --num_inference_steps 20 --file_name_render test_transformer_combos0 --seed 0 --pretrained_value_model runs/walker2d-medium-v2/value_1734398105 --checkpoint_value_model=180000 --pretrained_act_model runs/walker2d-medium-v2/1734402336 --checkpoint_act_model=799999 --pretrained_dyn_model runs/walker2d-medium-v2/1734404160 --checkpoint_dyn_model=799999 --n_episodes=5 --render_steps=2000

cheetah
python scripts/eval_hf_value_dmpc.py --env_name halfcheetah-medium-v2 --use-ema --num_inference_steps 20 --file_name_render test_transformer_combos0 --seed 0 --pretrained_value_model runs/halfcheetah-medium-v2/value_1734390656 --checkpoint_value_model=180000 --pretrained_act_model runs/halfcheetah-medium-v2/1734401044 --checkpoint_act_model=799999 --pretrained_dyn_model runs/halfcheetah-medium-v2/1734407834 --checkpoint_dyn_model=799999 --n_episodes=5 --render_steps=2000

## eval reward gen
python scripts/eval_hf_value_dmpc_height.py --env_name walker2d-medium-v2  --use-ema --num_inference_steps 20 --file_name_render height --seed 0 --pretrained_value_model runs/walker2d-medium-v2/value_1734398105 --checkpoint_value_model=180000 --pretrained_act_model runs/walker2d-medium-v2/1734402336 --checkpoint_act_model=799999 --pretrained_dyn_model runs/walker2d-medium-v2/1734404160 --checkpoint_dyn_model=799999 --render_steps=50 --target_height 0.9 --sigma2 0.0005

python scripts/eval_hf_value_dmpc_height.py --env_name halfcheetah-medium-v2 --use-ema --num_inference_steps 20 --file_name_render height_cheetah --seed 0 --pretrained_value_model runs/halfcheetah-medium-v2/value_1734390656 --checkpoint_value_model=180000 --pretrained_act_model runs/halfcheetah-medium-v2/1734401044 --checkpoint_act_model=799999 --pretrained_dyn_model runs/halfcheetah-medium-v2/1734407834 --checkpoint_dyn_model=799999 --render_steps=50 



## eval with discount 0.997
python scripts/eval_hf_value_dmpc.py --use-ema --num_inference_steps 20 --file_name_render test_transformer_combos0 --seed 0 --pretrained_value_model runs/hopper-medium-v2/value_1734128018 --checkpoint_value_model=180000 --pretrained_act_model runs/hopper-medium-v2/1734141113 --checkpoint_act_model=799999 --pretrained_dyn_model runs/hopper-medium-v2/1734140811 --checkpoint_dyn_model=799999



## train hf value
python scripts/train_hf_value.py --train_batch_size 64 --gradient_accumulation_steps 1 --use-ema --discount_factor 0.997 --num_train_timesteps 20 --seed 42



## eval_hf_value w/ wandb logging, would need to add value
python3 eval_hf_value.py --use-ema --scale 0.1 --num_inference_steps 20 --file_name_render test_transformer_s01 --pretrained_diff_model runs/hopper-medium-v2/1734063759 --checkpoint_diff_model 799999 --render_steps 1000 --n_episodes 2


# D-MPC Unet training

## dynamics model training for D-MPC UNet
python scripts/train_hf_dynamics_unet.py --env_id hopper-medium-v2 --train_batch_size 64 --horizon 32 --num_train_timesteps 20 --seed 0 --history 0 --n_train_steps 1000000 --checkpointing_freq 100000
python scripts/train_hf_dynamics_unet.py --env_id walker2d-medium-v2 --train_batch_size 64 --horizon 32 --num_train_timesteps 20 --seed 0 --history 0 --n_train_steps 1000000 --checkpointing_freq 100000
python scripts/train_hf_dynamics_unet.py --env_id halfcheetah-medium-v2 --train_batch_size 64 --horizon 32 --num_train_timesteps 20 --seed 0 --history 0 --n_train_steps 1000000 --checkpointing_freq 100000

## action model training for D-MPC UNet 
python scripts/train_hf_action_unet.py --env_id hopper-medium-v2 --train_batch_size 64 --horizon 32 --num_train_timesteps 20 --seed 0 --history 0 --n_train_steps 1000000 --checkpointing_freq 100000 --render_freq 0
python scripts/train_hf_action_unet.py --env_id walker2d-medium-v2 --train_batch_size 64 --horizon 32 --num_train_timesteps 20 --seed 0 --history 0 --n_train_steps 1000000 --checkpointing_freq 100000 --render_freq 0
python scripts/train_hf_action_unet.py --env_id halfcheetah-medium-v2 --train_batch_size 64 --horizon 32 --num_train_timesteps 20 --seed 0 --history 0 --n_train_steps 1000000 --checkpointing_freq 100000 --render_freq 0

## value function training for original Diffuser
python scripts/train_hf_value.py --env_id hopper-medium-v2 --train_batch_size 64 --horizon 32 --num_train_timesteps 20 --arch_type unet --seed 0
python scripts/train_hf_value.py --env_id walker2d-medium-v2 --train_batch_size 64 --horizon 32 --num_train_timesteps 20 --arch_type unet --seed 0
python scripts/train_hf_value.py --env_id halfcheetah-medium-v2 --train_batch_size 64 --horizon 32 --num_train_timesteps 20 --arch_type unet --seed 0

## evaluation
python scripts/eval_hf_value_dmpc_unet.py --env_id hopper-medium-v2 --pretrained_act_model runs/hopper-medium-v2/action_unet_1734465526  --checkpoint_act_model 799999 --pretrained_dyn_model runs/hopper-medium-v2/dynamics_unet_1734415099 --checkpoint_dyn_model 799999 --pretrained_value_model runs/hopper-medium-v2/value_1734390656 --checkpoint_value_model 180000 --seed 0 --num_inference_steps 20 --use_ema --planning_horizon 32 --n_episodes 5 --render_steps 1000
python scripts/eval_hf_value_dmpc_unet.py --env_id walker2d-medium-v2 --pretrained_act_model runs/walker2d-medium-v2/action_unet_1734466176  --checkpoint_act_model 799999 --pretrained_dyn_model runs/walker2d-medium-v2/dynamics_unet_1734415238 --checkpoint_dyn_model 799999 --pretrained_value_model runs/walker2d-medium-v2/value_1734390664 --checkpoint_value_model 180000 --seed 0 --num_inference_steps 20 --use_ema --planning_horizon 32 --n_episodes 5 --render_steps 1000
python scripts/eval_hf_value_dmpc_unet.py --env_id halfcheetah-medium-v2 --pretrained_act_model runs/halfcheetah-medium-v2/action_unet_1734466190  --checkpoint_act_model 799999 --pretrained_dyn_model runs/halfcheetah-medium-v2/dynamics_unet_1734415258 --checkpoint_dyn_model 799999 --pretrained_value_model runs/halfcheetah-medium-v2/value_1734390667 --checkpoint_value_model 180000 --seed 0 --num_inference_steps 20 --use_ema --planning_horizon 32 --n_episodes 5 --render_steps 1000


# Behavior cloning 

## training
python scripts/bc_d4rl.py --env_id hopper-medium-v2 --train_batch_size 256 --horizon 1 --seed 0 --checkpointing_freq 50000 --n_train_steps 1000000
python scripts/bc_d4rl.py --env_id walker2d-medium-v2 --train_batch_size 256 --horizon 1 --seed 0 --checkpointing_freq 50000 --n_train_steps 1000000
python scripts/bc_d4rl.py --env_id halfcheetah-medium-v2 --train_batch_size 256 --horizon 1 --seed 0 --checkpointing_freq 50000 --n_train_steps 1000000

## evaluation
python scripts/bc_d4rl.py --env_id hopper-medium-v2 --run_eval_only --n_episodes 5 --render --seed 0 --render_steps 1000 --pretrained_model runs/hopper-medium-v2/behavior-cloning_1734494380 --checkpoint_model 799999
python scripts/bc_d4rl.py --env_id walker2d-medium-v2 --run_eval_only --n_episodes 5 --render --seed 0 --render_steps 1000 --pretrained_model runs/walker2d-medium-v2/behavior-cloning_1734494500 --checkpoint_model 799999
python scripts/bc_d4rl.py --env_id halfcheetah-medium-v2 --run_eval_only --n_episodes 5 --render --seed 0 --render_steps 1000 --pretrained_model runs/halfcheetah-medium-v2/behavior-cloning_1734494553 --checkpoint_model 799999

