#!/bin/bash

source config.system.sh

INDEX_TARGET="42 88 93 96 110"
INDEX_TARGET="88" # Macaw
LOSS_TARGETS="mse vgg_per adv"
NUM_EPOCH=20

CUDA_VISIBLE_DEVICES=$GPUS python -W ignore train.py \
                    --vgg_target_layers 1 2 13 20 \
                    --no_save \
                    --loss_targets $LOSS_TARGETS \
                    --size_batch $SIZE_BATCH \
                    --interval_save_loss 10 \
                    --interval_save_train 10 \
                    --interval_save_test 10 \
                    --num_test_sample 8 \
                    --num_copy $NUM_COPY \
                    --index_target $INDEX_TARGET \
                    --num_epoch $NUM_EPOCH \
                    --path_log 'runs' \
                    --task_name 'wip_v1' \
                    --detail 'wip' 
