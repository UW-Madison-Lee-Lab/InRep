#!/bin/bash

data=$1
nclass=$2
epochs=$3

#####========= Training UnConditional  =========#####
python3 main.py \
    --data_type $data --num_classes $nclass\
    --gan_type 1 --repgan_phase 1\
    --resume 0\
    --nepochs $epochs --nsteps_save 50\
    --is_train
