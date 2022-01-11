#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3 python -W ignore train.py \
                                        --use_attention \
                                        --use_enhance \
                                        --coef_attention 1.4 \
                                        --task_name attention_aug \
                                        --detail "attention 64 and aug 1.4" 
                                                       
