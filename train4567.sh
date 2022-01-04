#!/bin/bash

CUDA_VISIBLE_DEVICES=4,5,6,7 python -W ignore train.py --port 12356
