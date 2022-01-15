source scripts/config  

for c in "11 14" "14 15" "15 11"; do
    python -W ignore app/colorize_intp.py --classes $c \
                                          --path_ckpt $PATH_CKPT \
                                          --epoch $EPOCH \
                                          --use_ema \
                                          --path_input resource/grays_intrp \
                                          --path_output exprs/intrp
done
