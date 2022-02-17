#!/bin/bash

set -e

# Must be set
EPOCH=9
DIM_F=16
NAME_TASK=ablation_fdim_16

PATH_CKPT=ckpts/$NAME_TASK
epoch=$(printf %03d $EPOCH)
dir_target="$(pwd)/results/${NAME_TASK}_e${epoch}"
dir_gt='gt_valid'

# INFERENCE
python -W ignore colorize.py \
    --path_ckpt $PATH_CKPT \
    --path_output ${dir_target}_lab \
    --epoch $EPOCH \
    --dim_f $DIM_F \
    --seed -1 \
    --iter_max 5000 \
    --use_ema 
