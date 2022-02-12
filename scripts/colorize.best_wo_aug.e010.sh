#!/bin/bash

set -e

# Must be set
EPOCH=10
DIM_F=16
NAME_TASK=best_wo_aug

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

# EXTRACT COMPLEX IMAGE
dir_target_cplx=${dir_target}_cplx
mkdir -p $dir_target_cplx
for i in $(cat cplx_img_ids.txt); do
    i=${dir_target}/$i.jpg
    if [ -f $i ]; then
        cp $i ${dir_target_cplx}
    else
        echo no
    fi
done
im-grid $dir_target_cplx

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
