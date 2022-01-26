#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3 python -W ignore train.py \
        --use_enhance \
        --coef_enhance 1.2 \
        --path_log 'runs_c1000' \
        --vgg_target_layers 1 2 13 14 \
        --size_batch 60 \
        --num_epoch 15 \
        --schedule_type 'linear' \
        --task_name 'linear_shedule' \
        --detail "Use linearly decreased learning rate scheduler and \
        slightly change the VGG target feature"
