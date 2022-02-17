#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1 python -W ignore train.py \
                    --port 12357 \
                    --vgg_target_layers 1 2 13 20 \
                    --size_batch 30 \
                    --num_epoch 10 \
                    --norm_type id \
                    --no_res
                    --schedule_type mult \
                    --path_log 'runs_c100' \
                    --task_name 'ablation_nonorm_nor' \
                    --index_target {0..99} \
                    --detail 'no_norm and no res' 
