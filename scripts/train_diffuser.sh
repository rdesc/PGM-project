#!/bin/bash
# Training script for running the diffuser (J)
# Define arrays for seeds and environment IDs
seeds=(0)
env_ids=("hopper-medium-v2" "halfcheetah-medium-v2" "walker2d-medium-v2")

# Function to run training for value functions
train_value_functions() {
  for env_id in "${env_ids[@]}"; do
    for seed in "${seeds[@]}"; do
      sbatch train train_hf_value.py \
        --train_batch_size=64 \
        --gradient_accumulation_steps=1 \
        --use-ema \
        --discount_factor=0.997 \
        --num_train_timesteps=20 \
        --seed="$seed" \
        --env_id="$env_id"
    done
  done
}

# Function to run training for diffuser models
train_diffuser_models() {
  for env_id in "${env_ids[@]}"; do
    for seed in "${seeds[@]}"; do
      sbatch train train_hf.py \
        --train_batch_size=64 \
        --gradient_accumulation_steps=1 \
        --num_train_timesteps=20 \
        --seed="$seed" \
        --env_id="$env_id" \
        --use_original_config \
        --weight_decay=0.0 \
        --horizon=32 \
        --n_train_steps=1000000 \
        --checkpointing_freq=100000 \
        --render_freq=100000 \
        --action_weight=10 \
        --no-cosine_warmup \
        --learning_rate=0.0002 \
        --mixed_precision=no
    done
  done
}

# Run the functions
echo "Starting training for value functions..."
train_value_functions

# echo "Starting training for diffuser models..."
# train_diffuser_models
