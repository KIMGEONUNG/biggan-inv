#!/bin/bash

set -e

dir_target="$(pwd)/results/lpips_low_feat_e011"
dir_gt='gt_valid'

# INFERENCE
python -W ignore colorize.py \
    --path_ckpt './ckpts/lpips_low_feat' \
    --path_output $dir_target \
    --epoch 11 \
    --use_ema 

# MAKE_GRID
im-grid $dir_target

# SET
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
