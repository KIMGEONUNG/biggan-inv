#!/bin/bash

# CUDA_VISIBLE_DEVICES=0,1,2,3 python -W ignore train.py \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 python -W ignore train.py \
                    --use_enhance \
                    --coef_enhance 1.2 \
                    --vgg_target_layers 1 2 13 20 \
                    --size_batch 160 \
                    --num_epoch 12 \
                    --schedule_type mult \
                    --dim_f 16 \
                    --path_log 'runs_c1000' \
                    --task_name 'best_full_batch_a' \
                    --detail "full batch" 
