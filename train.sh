#!/bin/bash

# CUDA_VISIBLE_DEVICES=0,1,2,3 python -W ignore train.py \
CUDA_VISIBLE_DEVICES=0 python -W ignore train.py \
                    --vgg_target_layers 1 2 13 20 \
                    --loss_mse \
                    --loss_lpips \
                    --loss_adv \
                    --size_batch 4 \
                    --index_target 42 88 93 96 110 \
                    --num_epoch 12 \
                    --num_copy 4 \
                    --path_log 'runs' \
                    --task_name 'wip' \
                    --detail 'wip' 
