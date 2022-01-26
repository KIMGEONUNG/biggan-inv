#!/bin/bash

# INFERENCE
python -W ignore colorize_real.py \
    --path_ckpt './ckpts/fix_lpips' \
    --epoch 11 \
    --use_ema 
