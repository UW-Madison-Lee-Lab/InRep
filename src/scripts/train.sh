#!/bin/bash
data=cifar10
gan_type=inrep
exp_mode=complexity
phase=1
resume=0
epochs=100
gan_class=0
label_ratio=0.1

python main.py \
    --data_type $data \
    --gan_type $gan_type --gan_class $gan_class\
    --exp_mode $exp_mode --phase $phase \
    --is_train --nepochs $epochs --resume $resume --nsteps_save 10 \

echo "Done CGAN"