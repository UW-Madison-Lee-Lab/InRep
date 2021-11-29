#!/bin/bash
data=$1
gan_type=$2
exp_mode=$3
phase=1
resume=0
epochs=300

python main.py \
    --data_type $data \
    --gan_type $gan_type --gan_class $gan_class\
    --exp_mode $exp_mode --phase $phase \
    --is_train --nepochs $epochs --resume $resume --nsteps_save 10 \

echo "Done CGAN"
python main.py --is_train -d tiny -g ugan -e 200 -r 1
python main.py -d cifar10 -g udecoder -p flow -v visual
python main.py --is_train -d tiny -g ugan -e 200 -r 1

python main.py -d cifar10 -g ganrep -v fid