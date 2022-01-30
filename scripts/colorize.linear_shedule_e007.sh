#!/bin/bash

set -e

EPOCH=7
epoch=$(printf %03d $EPOCH)
NAME_TASK=linear_shedule
PATH_CKPT=ckpts/$NAME_TASK

dir_target="$(pwd)/results/${NAME_TASK}_e${epoch}"
dir_gt='gt_valid'

# INFERENCE
python -W ignore colorize.py \
    --path_ckpt $PATH_CKPT \
    --path_output $dir_target \
    --epoch $EPOCH \
    --use_ema 

# MAKE_GRID
im-grid $dir_target

# Set
dir_target_m=${dir_target}_m
mkdir -p $dir_target_m

# FID
python -m pytorch_fid  $dir_target $dir_gt > $dir_target_m/fid.txt

# COLORFUL
echo $dir_target/*.jpg | xargs -n 1 | colorfullness > $dir_target_m/colorful.txt
cat $dir_target_m/colorful.txt | average > $dir_target_m/colorful_avg.txt

# ACC
dir_ln="$(pwd)/results/target"
rm $dir_ln -f
ln -s $dir_target $dir_ln
cal_acc ./results/for_acc > $dir_target_m/acc.txt
