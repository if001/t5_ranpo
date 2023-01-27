#!/bin/sh
python train.py \
       -o ./saved_model/gpt2_rnp_with_title \
       -d ./dataset/with_title \
       --max_data_len 20000 \
       --type gpt \
       --max_seq_len 256 \
       --batch 1 \
       --epoch 1
