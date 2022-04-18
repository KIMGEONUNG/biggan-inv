#!/bin/bash

source config.system.sh

# CUDA_VISIBLE_DEVICES=0,1,2,3 python -W ignore train.py \
CUDA_VISIBLE_DEVICES=$GPUS python -W ignore train.py \
                    --vgg_target_layers 1 2 13 20 \
                    --loss_mse \
                    --loss_lpips \
                    --loss_adv \
                    --size_batch $SIZE_BATCH \
                    --num_copy $NUM_COPY \
                    --index_target 42 88 93 96 110 \
                    --num_epoch 12 \
                    --path_log 'runs' \
                    --task_name 'wip' \
                    --detail 'wip' 
