#!/bin/bash

CUDA_VISIBLE_DEVICES=4,5 python -W ignore train.py \
                    --port 12356 \
                    --vgg_target_layers 1 2 13 20 \
                    --size_batch 30 \
                    --num_epoch 10 \
                    --schedule_type mult \
                    --dim_f 32 \
                    --path_log 'runs_c100' \
                    --task_name 'ablation_fdim_32' \
                    --index_target {0..99} \
                    --detail 'ablation, embd f32' 
