#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python -W ignore train.py \
                    --size_batch 16 \
                    --num_epoch 12 \
                    --schedule_type mult \
                    --dim_f 16 \
                    --path_log 'runs' \
                    --task_name 'MVP' \
                    --detail "no contents" 
