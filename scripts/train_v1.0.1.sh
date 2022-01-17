#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3 python -W ignore train.py \
        --use_enhance \
        --coef_enhance 1.2 \
        --path_log 'runs_c1000' \
        --vgg_target_layers 1 2 6 7 \
        --size_batch 60 \
        --task_name 'lpips_low_feat' \
        --detail "Previous exp,train_v1.0.0, high level feature, like 23, 
        invoke unnatural results. For the more accurate color inference, we 
        need more low level feature, so change the vgg_target_layers" 
