#!/bin/bash
data=$1
gan_type=$3
exp_mode=$4
nlabels=$5
nnoises=$6
c=$7
epochs=$8

python main.py \
    --data_type $data \
    --gan_type $gan_type\
    --gan_class $c\
    --exp_mode $exp_mode --label_ratio $nlabels --noise_ratio $nnoises\
    --nepochs $epochs --nsteps_save 10\
    --is_train --resume 0

echo "Done CGAN"
