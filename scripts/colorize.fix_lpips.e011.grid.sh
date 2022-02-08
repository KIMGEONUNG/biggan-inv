#!/bin/bash

set -e

dir_target="$(pwd)/results/fix_lpips_e011_grid_tmp"
dir_gt='gt_valid'

# INFERENCE
python -W ignore colorize_grid.py \
    --path_ckpt './ckpts/fix_lpips' \
    --path_output $dir_target \
    --epoch 11 \
    --use_square \
    --use_ema 
