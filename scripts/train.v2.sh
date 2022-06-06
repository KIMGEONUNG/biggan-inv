#!/bin/bash

source config.system.sh

INDEX_TARGET=$(echo {0..99})
LOSS_TARGETS="adv"
NUM_EPOCH=20

CUDA_VISIBLE_DEVICES=$GPUS python -W ignore train.py \
                    --vgg_target_layers 1 2 13 20 \
                    --loss_targets $LOSS_TARGETS \
                    --size_batch $SIZE_BATCH \
                    --interval_save_loss 10 \
                    --interval_save_train 100 \
                    --interval_save_test 100 \
                    --dim_encoder_c 128 \
                    --num_test_sample 15 \
                    --unaligned_sample \
                    --index_target $INDEX_TARGET \
                    --num_epoch $NUM_EPOCH \
                    --path_log 'runs' \
                    --task_name $(echo ${0##*/} | sed 's:.sh::' | sed 's:train.::') \
                    --detail 'wip' 
