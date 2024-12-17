#!/bin/bash
# Training script for running the diffuser (J)
# Define arrays for seeds and environment IDs
seeds=(0, 1, 2, 3, 4)
env_ids=("hopper-medium-v2" "halfcheetah-medium-v2" "walker2d-medium-v2")
declare -A diff_ids=( ["halfcheetah-medium-v2"]="1734391524" ["hopper-medium-v2"]="1734391515" ["walker2d-medium-v2"]="1734391523")
declare -A value_ids=( ["halfcheetah-medium-v2"]="value_1734390656" ["hopper-medium-v2"]="value_1734128018" ["walker2d-medium-v2"]="value_1734398105")

# Function to run training for diffuser models
train_diffuser_models() {
  for env_id in "${env_ids[@]}"; do
    for seed in "${seeds[@]}"; do
      sbatch train eval_hf_value.py \
        --seed="$seed" \
        --env_id="$env_id" \
        --num_inference_steps=20 \
        --n_guide_steps=0 \
        --scale=0.0 \
        --use_ema \
        --planning_horizon=32 \
        --pretrained_diff_model=runs/"$env_id"/"${diff_ids[$env_id]}" \
        --pretrained_value_model=runs/"$env_id"/"${value_ids[$env_id]}" \
        --checkpoint_diff_model=799999 \
        --checkpoint_value_model=180000 \
        --torch_compile 
    done
  done
}

# Run the functions
echo "Starting training for diffuser transformer models..."
train_diffuser_models

# echo "Starting training for diffuser models..."
# train_diffuser_models

# python scripts/eval_hf_value.py --file_name_render test_transformer_s01 --pretrained_diff_model runs/1734063759 --checkpoint_diff_model 799999
