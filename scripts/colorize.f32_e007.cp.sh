#!/bin/bash

set -e

# Must be set
EPOCH=7
DIM_F=32
NAME_TASK=f32

PATH_CKPT=ckpts/$NAME_TASK
epoch=$(printf %03d $EPOCH)
dir_target="$(pwd)/results/${NAME_TASK}_e${epoch}"
dir_gt='gt_valid'

# INFER REAL TARGET
dir_target_real=${dir_target}_real
mkdir -p $dir_target_real

python -W ignore colorize_real.py \
    --path_ckpt $PATH_CKPT \
    --path_input './resource/real_grays' \
    --path_output $dir_target_real \
    --epoch $EPOCH \
    --dim_f $DIM_F \
    --size_target 256 \
    --type_resize 'powerof' \
    --topk 5 \
    --use_ema 

# GROUPING
mv ${dir_target} ${dir_target}_val
mkdir -p ${dir_target}
mv ${dir_target}_* ${dir_target}
