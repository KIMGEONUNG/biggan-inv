#!/bin/bash

source config.system.sh

INDEX_TARGET=$(echo {0..99})
LOSS_TARGETS="adv"
NUM_EPOCH=20

CUDA_VISIBLE_DEVICES=$GPUS python -W ignore train.py \
                    --vgg_target_layers 1 2 13 20 \
                    --no_save \
                    --loss_targets $LOSS_TARGETS \
                    --size_batch $SIZE_BATCH \
                    --interval_save_loss 10 \
                    --interval_save_train 10 \
                    --interval_save_test 10 \
                    --dim_encoder_c 128 \
                    --chunk_size_z_e 0 \
                    --coef_wip 0.02 \
                    --num_test_sample 15 \
                    --index_target $INDEX_TARGET \
                    --num_epoch $NUM_EPOCH \
                    --path_log 'runs' \
                    --task_name $(echo ${0##*/} | sed 's:.sh::' | sed 's:train.::') \
                    --detail 'wip' 
