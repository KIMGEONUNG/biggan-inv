#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1 python -W ignore train.py \
                    --port 12358 \
                    --vgg_target_layers 1 2 13 20 \
                    --size_batch 30 \
                    --num_epoch 10 \
                    --no_dropout \
                    --schedule_type mult \
                    --dim_f 16 \
                    --path_log 'runs_c100' \
                    --task_name 'ablation_nodrop' \
                    --index_target {0..99} \
                    --detail 'ablation no dropout' 
