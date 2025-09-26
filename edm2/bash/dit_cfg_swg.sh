#!/bin/bash
####################################################################
# How to launch: 
# source .venv/bin/activate && bash bash/dit_cfg_swg.sh
####################################################################
export CUDA_VISIBLE_DEVICES=0,1,2,3
nproc_per_node=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

################################################################################################
# Change this:
base_out_dir="./samples/IN256/cond_DiT_CFG_SWG"
ref_path="./dataset_refs/img256.pkl"
################################################################################################


# CFG+ M-SWG with masking (best FDD)
# g_scale=0.6
# scale=0.5
# mask=1

# CFG+SWG (best FDD)
g_scale=0.7
scale=0.1
mask=0

torchrun --standalone --nproc_per_node=$nproc_per_node sample_dit.py --save_tar \
    --model DiT-XL/2 --num-fid-samples 50000 --g_scale $g_scale --per-proc-batch-size 32 \
    --cfg_crop 1 --crop_scale $scale --mask $mask \
    --sample-dir $base_out_dir --ref_path $ref_path --metrics fid fd_dinov2 ;

