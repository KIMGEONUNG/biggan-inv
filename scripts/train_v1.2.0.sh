#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3 python -W ignore train.py \
                    --use_enhance \
                    --coef_enhance 1.2 \
                    --vgg_target_layers 1 2 13 20 \
                    --size_batch 60 \
                    --num_epoch 15 \
                    --schedule_type mult \
                    --dim_f 32 \
                    --path_log 'runs_c1000' \
                    --task_name 'f32' \
                    --detail "use 32 x 32 embeding space" 
