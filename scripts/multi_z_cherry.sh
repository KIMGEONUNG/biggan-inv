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
    --targets 3830 16191 16358 20557 21273 20564 45630 35906 26932 32940 \
    --z_std 5 \
    --num_z 20 \
    --use_shuffle \
    --seed -1 \
    --use_ema 

pushd results/fix_lpips_e011_zs_cherry
ls | im-ccrop --short -r
popd


## Candidates
# 3830 16191 16358 20557 21273 20564
