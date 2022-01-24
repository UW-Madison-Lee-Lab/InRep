#!/bin/bash
resume=1
data=cifar10
gan_type=inrep
exp_mode=asymnoise
epochs=100


python main.py \
    --data_type $data\
    --gan_type $gan_type\
    --exp_mode $exp_mode\
    --is_train --nepochs $epochs --resume $resume --nsteps_save 10 \
    --eval_mode intrafid

echo "Done CGAN"