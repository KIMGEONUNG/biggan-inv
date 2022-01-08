#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3 python -W ignore train.py --retrain\
                                                       --retrain_epoch 4
