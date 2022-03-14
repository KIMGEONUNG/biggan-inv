#!/bin/bash

set -e

# Must be set
EPOCH=9
DIM_F=16
NAME_TASK=ablation_nodrop

PATH_CKPT=ckpts/$NAME_TASK
epoch=$(printf %03d $EPOCH)
dir_target="$(pwd)/results/${NAME_TASK}_e${epoch}"
dir_gt='gt_valid'

# INFERENCE
python -W ignore colorize.py \
    --path_ckpt $PATH_CKPT \
    --path_output $dir_target \
    --epoch $EPOCH \
    --dim_f $DIM_F \
    --iter_max 5000 \
    --use_rgb \
    --use_ema 

# MAKE_GRID
im-grid $dir_target

# GROUPING
mv ${dir_target} ${dir_target}_val
mkdir -p ${dir_target}
mv ${dir_target}_* ${dir_target}
