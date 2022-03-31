#!/bin/bash

CUDA_VISIBLE_DEVICES=4,5,6,7 python -W ignore train.py \
                    --port 12356 \
                    --vgg_target_layers 1 2 13 20 \
                    --size_batch 60 \
                    --num_epoch 15 \
                    --schedule_type mult \
                    --dim_f 16 \
                    --path_log 'runs' \
                    --task_name 'MVP_subset_baseline' \
                    --detail 'no contents' 
