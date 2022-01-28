source scripts/config  

python -W ignore ./app/colorize_opt_f.py --class_id 15 \
                                         --path_ckpt $PATH_CKPT \
                                         --epoch $EPOCH \
                                         --use_ema \
                                         --num_iter 200 \
                                         --optimizer adam \
                                         --loss mse \
                                         --num_layer_f 3 \
                                         --vgg_target_layers 1 2 6 7 \
                                         --feat_mask \
                                         --path_input resource/grays_opt/ILSVRC2012_val_00025035.JPEG \
                                         --path_input_mask resource/masks \
                                         --path_output exprs/opt_f
