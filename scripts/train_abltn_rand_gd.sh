#!/bin/bash

CUDA_VISIBLE_DEVICES=4,5 python -W ignore train.py \
                    --port 12358 \
                    --vgg_target_layers 1 2 13 20 \
                    --size_batch 30 \
                    --num_epoch 10 \
                    --schedule_type mult \
                    --dim_f 16 \
                    --no_pretrained_g \
                    --no_pretrained_d \
                    --path_log 'runs_c100' \
                    --task_name 'ablation_rand_gd' \
                    --index_target {0..99} \
                    --detail 'random init generator & discriminator' 
