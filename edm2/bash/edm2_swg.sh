#!/bin/bash
####################################################################

# How to launch: 
# source .venv/bin/activate && bash bash/edm2_swg.sh
####################################################################
export CUDA_VISIBLE_DEVICES=0,1
nproc_per_node=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
source ./bash/gpu_utils.sh  # path for polling and check gpu space
base_out_dir="./samples"
batch=100
resolution=512
metrics='fid,fd_dinov2'
swg_sizes="40" # crop size, k in the paper
swg_steps="2" # crops steps per dimension for SWG: swg_steps^2 = N in the paper
seeds="0-49999"
g_interval="none"
ref_path="https://nvlabs-fi-cdn.nvidia.com/edm2/dataset-refs/img512.pkl"
model_size="s"
net_pos_pkl="https://nvlabs-fi-cdn.nvidia.com/edm2/raw-snapshots/edm2-img512-s/edm2-img512-s-2147483-0.100.pkl"
g_method='swg' 
poll=0 # whether to poll gpu space or not

for g_weight in "0.2" "1.1"   
do
    g_weight_name=$(echo $g_weight | tr ',' '_') # Careful!!! This is a hack to replace the comma with underscore
    outname="/IN${resolution}/${model_size}/MSWG/${g_weight_name}/samples/"
    outdir="$base_out_dir$outname"
    echo "Running with weight $g_weight, g_method $g_method, seeds $seeds, g_interval $g_interval"

    params="--net_pos_pkl $net_pos_pkl --outdir $outdir \
        --seeds $seeds --subdirs --batch $batch --metrics $metrics --ref_path $ref_path \
        --g_weight $g_weight --g_method $g_method --swg_sizes $swg_sizes --swg_steps $swg_steps --g_interval $g_interval" ;

    echo $params
    if [ "$poll" -eq 1 ]; then
        poll_and_launch "$nproc_per_node" "$params" 20m
    else
        torchrun --standalone --nproc_per_node="$nproc_per_node" sample_edm2.py $params
    fi

done


