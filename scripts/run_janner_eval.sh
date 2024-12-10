#!/bin/bash

# Define batches of commands
batch1=(
# hopper-medium-v2
"python scripts/plan_guided.py --diffusion_epoch 800000 --value_epoch 160000 --loadbase logs/pretrained --seed 0 --scale 10 --suffix scale_10_no_ssr --dataset hopper-medium-v2 --batch_size 1"
"python scripts/plan_guided.py --diffusion_epoch 800000 --value_epoch 160000 --loadbase logs/pretrained --seed 0 --scale 1.0 --suffix scale_1.0_no_ssr --dataset hopper-medium-v2 --batch_size 1"
"python scripts/plan_guided.py --diffusion_epoch 800000 --value_epoch 160000 --loadbase logs/pretrained --seed 0 --scale 0.1 --suffix scale_0.1_no_ssr --dataset hopper-medium-v2 --batch_size 1"
"python scripts/plan_guided.py --diffusion_epoch 800000 --value_epoch 160000 --loadbase logs/pretrained --seed 0 --scale 0.001 --suffix scale_0.001_no_ssr --dataset hopper-medium-v2 --batch_size 1"
"python scripts/plan_guided.py --diffusion_epoch 800000 --value_epoch 160000 --loadbase logs/pretrained --seed 0 --scale 0.0 --suffix scale_0.0_no_ssr --dataset hopper-medium-v2 --batch_size 1"

)
batch2=(
# walker2d-medium-replay-v2
"python scripts/plan_guided.py --diffusion_epoch 600000 --value_epoch 160000 --loadbase logs/pretrained --seed 0 --scale 10 --suffix scale_10_no_ssr --dataset walker2d-medium-replay-v2 --batch_size 1"
"python scripts/plan_guided.py --diffusion_epoch 600000 --value_epoch 160000 --loadbase logs/pretrained --seed 0 --scale 1.0 --suffix scale_1.0_no_ssr --dataset walker2d-medium-replay-v2 --batch_size 1"
"python scripts/plan_guided.py --diffusion_epoch 600000 --value_epoch 160000 --loadbase logs/pretrained --seed 0 --scale 0.1 --suffix scale_0.1_no_ssr --dataset walker2d-medium-replay-v2 --batch_size 1"
"python scripts/plan_guided.py --diffusion_epoch 600000 --value_epoch 160000 --loadbase logs/pretrained --seed 0 --scale 0.001 --suffix scale_0.001_no_ssr --dataset walker2d-medium-replay-v2 --batch_size 1"
"python scripts/plan_guided.py --diffusion_epoch 600000 --value_epoch 160000 --loadbase logs/pretrained --seed 0 --scale 0.0 --suffix scale_0.0_no_ssr --dataset walker2d-medium-replay-v2 --batch_size 1"

)
batch3=(
# halfcheetah-medium-expert-v2
"python scripts/plan_guided.py --diffusion_epoch 800000 --value_epoch 160000 --loadbase logs/pretrained --seed 0 --scale 10 --suffix scale_10_no_ssr --dataset halfcheetah-medium-expert-v2 --batch_size 1"
"python scripts/plan_guided.py --diffusion_epoch 800000 --value_epoch 160000 --loadbase logs/pretrained --seed 0 --scale 1.0 --suffix scale_1.0_no_ssr --dataset halfcheetah-medium-expert-v2 --batch_size 1"
"python scripts/plan_guided.py --diffusion_epoch 800000 --value_epoch 160000 --loadbase logs/pretrained --seed 0 --scale 0.1 --suffix scale_0.1_no_ssr --dataset halfcheetah-medium-expert-v2 --batch_size 1"
"python scripts/plan_guided.py --diffusion_epoch 800000 --value_epoch 160000 --loadbase logs/pretrained --seed 0 --scale 0.001 --suffix scale_0.001_no_ssr --dataset halfcheetah-medium-expert-v2 --batch_size 1"
"python scripts/plan_guided.py --diffusion_epoch 800000 --value_epoch 160000 --loadbase logs/pretrained --seed 0 --scale 0.0 --suffix scale_0.0_no_ssr --dataset halfcheetah-medium-expert-v2 --batch_size 1"

)

# Function to run a batch of commands in parallel
run_batch() {
    echo "Starting batch $1..."
    local batch=("${!2}")  # Get the batch array passed as an argument

    for cmd in "${batch[@]}"; do
        echo "Running: $cmd"
        $cmd &  # Run the command in the background
    done

    wait  # Wait for all background processes to finish
    echo "Batch $1 complete."
    echo
}

# Run batches
run_batch 1 batch1[@]
run_batch 2 batch2[@]
run_batch 3 batch3[@]

echo "All batches completed."