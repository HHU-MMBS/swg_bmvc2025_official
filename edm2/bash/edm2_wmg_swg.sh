#!/bin/bash
####################################################################
# How to launch: 
# source .venv/bin/activate && bash bash/edm2_wmg_swg.sh
####################################################################
export CUDA_VISIBLE_DEVICES=0,1,2,3 
nproc_per_node=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
source ./bash/gpu_utils.sh  # path for polling and check gpu space
base_out_dir="./samples"
batch=256
resolution=512
metrics='fid,fd_dinov2'
seeds="0-49999"
g_interval="none"
swg_sizes="40" # crop size, k in the paper
swg_steps="2" 
ref_path="https://nvlabs-fi-cdn.nvidia.com/edm2/dataset-refs/img512.pkl"
model_size="s"
g_method='wmg-swg' 
negative='RT' # RC, RCT
poll=1 # whether to poll gpu space or not
####################################################################

if [ "$negative" == "RT" ]; then
    # Reduce training time only (RT)
    if [ "$model_size" == "s" ]; then
        net_pos_pkl="https://nvlabs-fi-cdn.nvidia.com/edm2/raw-snapshots/edm2-img512-s/edm2-img512-s-2147483-0.100.pkl"
        net_neg_pkl="https://nvlabs-fi-cdn.nvidia.com/edm2/raw-snapshots/edm2-img512-s/edm2-img512-s-0134217-0.100.pkl" # S,T/16
    elif [ "$model_size" == "xxl" ]; then
        net_pos_pkl="https://nvlabs-fi-cdn.nvidia.com/edm2/raw-snapshots/edm2-img512-xxl/edm2-img512-xxl-0939524-0.100.pkl"
        net_neg_pkl="https://nvlabs-fi-cdn.nvidia.com/edm2/raw-snapshots/edm2-img512-xl/edm2-img512-xxl-0268435-0.100.pkl" # XXL,T/3.5
    fi
elif [ "$negative" == "RCT" ]; then
    # Reduce training time and capacity (RCT)
    if [ "$model_size" == "s" ]; then
        net_pos_pkl="https://nvlabs-fi-cdn.nvidia.com/edm2/raw-snapshots/edm2-img512-s/edm2-img512-s-2147483-0.100.pkl"
        net_neg_pkl="https://nvlabs-fi-cdn.nvidia.com/edm2/raw-snapshots/edm2-img512-xs/edm2-img512-xs-0134217-0.100.pkl" # XS,T/16
    elif [ "$model_size" == "xxl" ]; then
        net_pos_pkl="https://nvlabs-fi-cdn.nvidia.com/edm2/raw-snapshots/edm2-img512-xxl/edm2-img512-xxl-0939524-0.100.pkl"
        net_neg_pkl="https://nvlabs-fi-cdn.nvidia.com/edm2/raw-snapshots/edm2-img512-xl/edm2-img512-m-0268435-0.100.pkl" # M,T/3.5
    fi
else
    echo "Invalid negative"
    exit 1
fi

for g_weight in  "1.2,0.25" 
do
    g_weight_save=$(echo $g_weight | tr ',' '_')
    outname="/IN${resolution}/${model_size}/${negative}_SWG/${g_weight_save}/samples/"
    outdir="$base_out_dir$outname"
    # create the output directory if not exist
    mkdir -p $outdir
    echo $outdir
    
    echo "Running $negative with weight $g_weight, g_method $g_method, seeds $seeds, g_interval $g_interval"
    params="--net_pos_pkl $net_pos_pkl --net_neg_pkl $net_neg_pkl \
        --outdir $outdir  --swg_sizes $swg_sizes --swg_steps $swg_steps \
        --seeds $seeds --subdirs --batch $batch --metrics $metrics \
        --ref_path $ref_path --g_weight $g_weight --g_method $g_method \
        --g_interval $g_interval --use_wandb 0" ;
    
    if [ "$poll" -eq 1 ]; then
        poll_and_launch "$nproc_per_node" "$params" 10m
    else
        torchrun --standalone --nproc_per_node="$nproc_per_node" sample_edm2.py $params
    fi
done
