#!/bin/bash

# CUDA_VISIBLE_DEVICES=0,1,2,3 python -W ignore train.py \
CUDA_VISIBLE_DEVICES=0 python -W ignore train.py \
                    --vgg_target_layers 1 2 13 20 \
                    --size_batch 16 \
                    --num_epoch 12 \
                    --path_log 'runs' \
                    --task_name 'wip' \
                    --detail 'wip' 
