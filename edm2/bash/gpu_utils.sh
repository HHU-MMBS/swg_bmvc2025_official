#!/bin/bash
nproc_per_node=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

# Function to retrieve GPUs sorted by free memory in descending order
function get_gpus_sorted_by_memory() {
    local gpu_list=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t',' -k2 -nr | cut -d',' -f1)
    echo $gpu_list
}

# Function to check if GPU is free
function is_gpu_free() {
    local gpu_id=$1
    local gpu_memory=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $gpu_id)
    # Check if memory is greater than a threshold (e.g., 100 MB)
    echo "GPU $gpu_id has $gpu_memory MB free memory"
    if [ $gpu_memory -gt 30000 ]; then
        return 0  # GPU is free
    else
        return 1  # GPU is not free
    fi
}
            
# New function to poll for free GPU and launch script
function poll_and_launch() {
    local nproc_per_node="$1"  # Number of processes per node
    local params="$2"          # Parameters for the Python script
    local poll_interval="${3:-10m}"  # Polling interval, default to 10 minutes

    while true; do
        echo "Checking for free GPUs..."
        # Get GPUs sorted by free memory
        sorted_gpus=$(get_gpus_sorted_by_memory)
        
        for gpu_id in $sorted_gpus; do
            if is_gpu_free "$gpu_id"; then
                echo "GPU $gpu_id is free. Launching Python script..."
                echo "Running with parameters: $params"
                
                torchrun --standalone --nproc_per_node="$nproc_per_node" sample_edm2.py $params
                
                # Exit the function after successful launch
                return 0
            fi
        done
        
        echo "No free GPU available. Polling again in $poll_interval..."
        sleep "$poll_interval"
    done
}
