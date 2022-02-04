#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3 python -W ignore train.py \
                    --use_enhance \
                    --coef_enhance 1.2 \
                    --vgg_target_layers 1 2 13 20 \
                    --size_batch 60 \
                    --num_epoch 10 \
                    --schedule_type mult \
                    --dim_f 8 \
                    --path_log 'runs_c100' \
                    --task_name 'ablation_fdim_08' \
                    --index_target {0..99} \
                    --detail 'ablation, embd f8' 
