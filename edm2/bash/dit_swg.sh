#!/bin/bash
####################################################################
# How to launch: 
# source .venv/bin/activate && bash bash/dit_swg.sh
####################################################################
export CUDA_VISIBLE_DEVICES=0,1,2,3
nproc_per_node=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
num_samples=50000 # switch to 50000 for FID/FDD evaluation
################################################################################################
# Change this:
base_out_dir="./samples/IN256/cond_DiT_SWG"
ref_path="./dataset_refs/img256.pkl"
################################################################################################

# Record the start time in seconds and nanoseconds
start_time_sec=$(date +%s)
start_time_nsec=$(date +%N)

######################################################################
# M-SWG with masking with w=0.5 (best FID) and w=1.5 (best FDD)
mask=1
for scale in  0.5 # 1.5
do
    torchrun --standalone --nproc_per_node=$nproc_per_node sample_dit.py \
        --model DiT-XL/2 --num-fid-samples $num_samples --g_scale $scale --per-proc-batch-size 32 \
        --crop 1 --mask $mask  --sample-dir $base_out_dir --ref_path $ref_path --save_tar --metrics fid fd_dinov2
done
######################################################################


######################################################################
# SWG without masking with w=0.5
# mask=0
# scale=0.25 # 0.5 
# torchrun --standalone --nproc_per_node=$nproc_per_node sample_dit.py \
#     --model DiT-XL/2 --num-fid-samples $num_samples --g_scale $scale --per-proc-batch-size 16 \
#     --crop 1 --mask $mask  --sample-dir $base_out_dir --ref_path $ref_path --save_tar --metrics fid fd_dinov2 ;
######################################################################


# Record the end time in seconds and nanoseconds
end_time_sec=$(date +%s)
end_time_nsec=$(date +%N)

# Calculate the time difference in seconds
elapsed_sec=$(echo "$end_time_sec - $start_time_sec" | bc)
elapsed_nsec=$(echo "($end_time_nsec - $start_time_nsec)/1000000000" | bc -l)

# Total elapsed time in seconds with nanosecond precision
total_elapsed_time=$(echo "$elapsed_sec + $elapsed_nsec" | bc -l)

# Convert the total elapsed time into minutes without rounding
elapsed_minutes=$(echo "$total_elapsed_time / 60" | bc -l)

# Print results
echo "Elapsed time in seconds: $total_elapsed_time"
echo "Elapsed time in minutes (fractional): $elapsed_minutes"
