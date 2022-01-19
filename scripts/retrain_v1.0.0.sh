#!/bin/bash

CUDA_VISIBLE_DEVICES=4,5,6,7 python -W ignore train.py \
            --path_log 'runs_c1000' \
            --task_name fix_lpips \
            --use_enhance \
            --coef_enhance 1.2 \
            --vgg_target_layers 1 2 13 20 \
            --retrain \
            --retrain_epoch 4 \
            --port 12356 \
            --detail "fix the lpips_loss, retrain from 4" 

