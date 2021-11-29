#!/bin/bash

data=$1
nclass=$2
gan_type=1

exp_mode=1
nlabels=1.0
nnoises=0

# start=$1
# end=$2

epochs=200
c=-1
# for c in $(seq $start $end)
# do
bash scripts/cgan.sh $data $nclass $gan_type $exp_mode $nlabels $nnoises $c $epochs 2>&1 | tee ../logs/exp"$exp_mode"_d"$data"-"$nclass"_g"$gan_type"_l"$nlabels"_n"$nnoises"_c"$c".out
# done