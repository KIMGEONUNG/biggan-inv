#!/bin/bash
source scripts/config

set -e

epoch=$(printf %03d $EPOCH)

dir_target="$(pwd)/results/${NAME_TASK}_e${epoch}_zs_cherry"
dir_gt='gt_valid'

# INFERENCE
python -W ignore app/colorize_zs_cherry.py \
    --path_ckpt './ckpts/fix_lpips' \
    --path_output $dir_target \
    --epoch $EPOCH \
    --targets 456{0..4}{0..9} \
    --z_std 3 \
    --num_z 10 \
    --use_shuffle \
    --use_ema 


## Candidates
#3830 16191 16358 20557


## Best
# 20564

