source scripts/config  

python -W ignore ./app/colorize_opt_cp.py --class_id 15 \
                                          --path_ckpt $PATH_CKPT \
                                          --epoch $EPOCH \
                                          --use_ema \
                                          --num_iter 100 \
                                          --optimizer adam \
                                          --loss feat_vgg \
                                          --path_input resource/grays_opt/ILSVRC2012_val_00025035.JPEG \
                                          --path_input_mask resource/masks \
                                          --path_output exprs/opt_cp
