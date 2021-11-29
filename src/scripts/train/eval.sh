#!/bin/bash

data=$1
nclass=$2
gan_type=$3

exp_mode=$4
nlabels=$5
nnoises=$6

eval_mode=$7
epochs=$8

python main.py \
    --data_type $data --num_classes $nclass\
    --gan_type $gan_type --repgan_phase 1 --gan_class -1\
    --exp_mode $exp_mode --eval_mode $eval_mode --label_ratio $nlabels --noise_ratio $nnoises\
    --nepochs $epochs --nsteps_save 100

echo "Done evaluation"
