#!/bin/bash

# INFERENCE
python -W ignore colorize_real.py \
    --path_ckpt './ckpts/fix_lpips' \
    --path_input './resource/real_grays' \
    --epoch 11 \
    --size_target 256 \
    --type_resize 'powerof' \
    --topk 5 \
    --use_ema 
