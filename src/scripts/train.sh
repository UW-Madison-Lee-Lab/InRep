#!/bin/bash

label_ratio=$1
resume=$2
data=cifar10
gan_type=inrep
exp_mode=complexity
gan_class=0
phase=1
epochs=100


python main.py \
    --data_type $data -l $label_ratio\
    --gan_type $gan_type\
    --exp_mode $exp_mode --phase $phase \
    --is_train --nepochs $epochs --resume $resume --nsteps_save 10 \
    --eval_mode intrafid  --gan_class 0
    

echo "Done CGAN"