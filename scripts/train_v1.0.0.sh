#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3 python -W ignore train.py \
                                        --use_enhance \
                                        --coef_enhance 1.2 \
                                        --task_name fix_lpips \
                                        --detail "fix the lpips_loss" 
                                                       
