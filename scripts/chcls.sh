source scripts/config  

python -W ignore ./app/colorize_chcls.py --classes 15 11 14 88 100 \
                                         --path_ckpt $PATH_CKPT \
                                         --epoch $EPOCH \
                                         --path_input resource/grays_chcls \
                                         --path_output exprs/chcls
