#!/bin/bash
data=$1
nclass=$2
gan_type=$3

exp_mode=$4
nlabels=$5
sample_size=$6

epochs=$7
start=$8
end=$9

#####========= Training Conditional  =========#####
if [ $exp_mode -eq 1 ]
then
  nlabels=$sample_size
  nnoises=0
else
  nnoises=$sample_size
fi
#running
for c in $(seq $start $end)
do
  python main.py \
      --data_type $data --num_classes $nclass\
      --gan_type $gan_type\
      --repgan_phase 2 --gan_class $c\
      --exp_mode $exp_mode --label_ratio $nlabels --noise_ratio $nnoises\
      --resume 0\
      --nepochs $epochs --nsteps_save 10\
      --is_train
done

echo "Done CGAN"
