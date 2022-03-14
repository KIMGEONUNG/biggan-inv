#!/bin/bash

# # INFERENCE
# python -W ignore colorize_batch.py \
#     --path_ckpt './ckpts/fix_lpips' \
#     --path_output results_train \
#     --path_dataset ./imgnet/train \
#     --epoch 11 \
#     --size_batch 100 \
#     --use_ema 

# INFERENCE
python -W ignore colorize_batch.py \
    --path_ckpt './ckpts/best_wo_aug/' \
    --path_output results_train_noaug \
    --path_dataset ./imgnet/train \
    --epoch 10 \
    --size_batch 100 \
    --use_ema 
