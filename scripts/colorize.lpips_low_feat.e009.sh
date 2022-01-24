#!/bin/bash

dir_target='./results/lpips_low_feat_e009'
dir_gt='gt_valid'

# INFERENCE
python -W ignore colorize.py \
    --path_ckpt './ckpts/lpips_low_feat' \
    --path_output $dir_target \
    --epoch 9 \
    --use_ema 
#
dir_target_m=${dir_target}_m
mkdir -p $dir_target_m

# FID
python -m pytorch_fid  $dir_target $dir_gt > $dir_target_m/fid.txt

# COLORFUL
echo $dir_target/*.jpg | xargs -n 1 | colorfullness > $dir_target_m/colorful.txt
cat $dir_target_m/colorful.txt | average > $dir_target_m/colorful_avg.txt
