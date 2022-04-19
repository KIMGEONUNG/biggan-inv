#!/bin/bash

source config.system.sh

INDEX_TARGET="42 88 93 96 110"
INDEX_TARGET="88" # Macaw
LOSS_TARGETS="mse vgg_per adv wip"
NUM_EPOCH=20

SIZE_BATCH=$(($SIZE_BATCH * $NUM_COPY))
NUM_COPY=1

CUDA_VISIBLE_DEVICES=$GPUS python -W ignore train.py \
                    --vgg_target_layers 1 2 13 20 \
                    --no_save \
                    --loss_targets $LOSS_TARGETS \
                    --eval_targets 'color_scatter_score' \
                    --size_batch $SIZE_BATCH \
                    --interval_save_loss 10 \
                    --interval_save_train 10 \
                    --interval_save_test 10 \
                    --coef_wip 0.02 \
                    --num_test_sample 20 \
                    --num_copy $NUM_COPY \
                    --index_target $INDEX_TARGET \
                    --num_epoch $NUM_EPOCH \
                    --path_log 'runs' \
                    --task_name $(echo ${0##*/} | sed 's:.sh::' | sed 's:train.::') \
                    --detail 'wip' 
