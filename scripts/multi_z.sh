#!/bin/bash
source scripts/config

set -e

epoch=$(printf %03d $EPOCH)

dir_target="$(pwd)/results/${NAME_TASK}_e${epoch}_zs"
dir_gt='gt_valid'

# INFERENCE
python -W ignore app/colorize_zs.py \
    --path_ckpt './ckpts/fix_lpips' \
    --path_output $dir_target \
    --epoch 11 \
    --z_std 2 \
    --num_z 5 \
    --use_shuffle \
    --use_ema 
