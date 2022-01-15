source scripts/config  

python -W ignore ./app/colorize_opt_c.py --class_id 15 \
                                         --path_ckpt $PATH_CKPT \
                                         --epoch $EPOCH \
                                         --use_ema \
                                         --loss feat_vgg \
                                         --path_input resource/grays_opt/ILSVRC2012_val_00025035.JPEG \
                                         --path_input_mask resource/masks \
                                         --path_output exprs/opt_c
