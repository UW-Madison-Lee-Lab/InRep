#!/bin/bash

resume=$1
data=imagenet_carn
epochs=100

python main.py \
    --data_type $data\
    --gan_type ugan\
    --is_train --nepochs $epochs --resume $resume --nsteps_save 10 

echo "Done UGAN"