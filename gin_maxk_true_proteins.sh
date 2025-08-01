#!/bin/bash

# Script to complete missing GIN maxk_true experiments for ogbn-proteins
echo "=== Running missing GIN maxk_true experiments for ogbn-proteins ==="
echo "Starting at: $(date)"

# Create logs directory if it doesn't exist
mkdir -p logs

run_gin_maxk_true_experiment() {
    local dataset=$1 model=$2 k=$3 gpu=${4:-0}
    local exp_id="${dataset}_${model}_k${k}_maxk_true"
    
    # Check if log file already exists
    if [ -f "logs/${exp_id}.log" ]; then
        echo "SKIPPING: ${exp_id}.log already exists"
        return 0
    fi
    
    local cmd="python maxk_gnn_integrated.py --dataset $dataset --model $model --maxk $k --gpu $gpu --use_maxk_kernels"
    
    export CUDA_VISIBLE_DEVICES=$gpu
    
    echo "Starting: $exp_id"
    echo "Command: $cmd"
    
    # Run with timeout, redirect output to log file
    timeout 7200 $cmd > "logs/${exp_id}.log" 2>&1
    local exit_code=$?
    
    if [ $exit_code -eq 124 ]; then
        echo "TIMEOUT: $exp_id after 2 hours"
    elif [ $exit_code -eq 0 ]; then
        echo "SUCCESS: $exp_id completed"
    else
        echo "FAILED: $exp_id with exit code $exit_code"
    fi
    
    echo "Finished: $exp_id"
    echo "----------------------------------------"
    sleep 3  # Brief pause between experiments
}

# Dataset and model constants
DATASET="ogbn-proteins"
MODEL="gin"
K_VALUES=(4 8 16 32 64 128 256)

echo "Running GIN maxk_true experiments for k values: ${K_VALUES[*]}"
echo "Total experiments to run: ${#K_VALUES[@]} (including boundary tests k=4,8,256)"
echo ""

# Run all missing GIN maxk_true experiments
current=0
for k in "${K_VALUES[@]}"; do
    current=$((current + 1))
    echo "Progress: $current/${#K_VALUES[@]}"
    run_gin_maxk_true_experiment "$DATASET" "$MODEL" "$k" 0
done

echo ""
echo "=== Verification ==="
echo "Checking for GIN maxk_true experiments:"

for k in "${K_VALUES[@]}"; do
    exp_id="${DATASET}_${MODEL}_k${k}_maxk_true"
    if [ -f "logs/${exp_id}.log" ]; then
        echo "✓ Found: ${exp_id}.log"
    else
        echo "✗ Missing: ${exp_id}.log"
    fi
done

echo ""
echo "All GIN maxk_true experiments for ogbn-proteins:"
ls -la logs/ogbn-proteins_gin_*_maxk_true.log 2>/dev/null || echo "No GIN maxk_true logs found"

echo ""
echo "=== Script completed at: $(date) ==="
echo "Missing GIN maxk_true experiments for ogbn-proteins have been executed."
