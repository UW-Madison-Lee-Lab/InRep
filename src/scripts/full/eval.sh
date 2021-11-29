#!/bin/bash

data=$1
nclass=$2
gan_type=$3
exp_mode=$4
nlabels=$5
sample_size=$6
eval_mode=$7
epochs=$8


  # runnning
if [[ $exp_mode -eq 1 ]]
then
  nlabels=$sample_size
  nnoises=0
else
  nnoises=$sample_size
fi

python main.py \
        --data_type $data --num_classes $nclass\
        --gan_type $gan_type --repgan_phase 2 --gan_class -1\
        --exp_mode $exp_mode --eval_mode $eval_mode --label_ratio $nlabels --noise_ratio $nnoises\
        --nepochs $epochs --resume 0 --nsteps_save 100

echo "Done evaluation"
