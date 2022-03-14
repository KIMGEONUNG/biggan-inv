#!/bin/bash

res=512
res=256

# INFERENCE
python -W ignore colorize_real.py \
    --path_ckpt './ckpts/fix_lpips' \
    --path_input './resource/real_grays' \
    --epoch 11 \
    --size_target $res \
    --type_resize 'powerof' \
    --topk 5 \
    --seed -1 \
    --use_ema 
