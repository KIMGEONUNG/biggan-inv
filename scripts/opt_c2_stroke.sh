source scripts/config  

python -W ignore ./app/colorize_opt_c2_stroke.py --class_id 15 \
                                          --path_ckpt $PATH_CKPT \
                                          --epoch $EPOCH \
                                          --use_ema \
                                          --num_iter 300 \
                                          --optimizer adam \
                                          --loss mse \
                                          --lr 1e-3 \
                                          --path_input resource/grays_opt/ILSVRC2012_val_00025035.JPEG \
                                          --path_input_mask resource/masks_stroke \
                                          --path_output exprs/opt_c2_stroke
