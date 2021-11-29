#!/bin/bash

data=$1
gan_type=$2

exp_mode=1
nlabels=1
nnoises=0
c=0

epochs=300

bash scripts/cgan.sh $data $nclass $gan_type $exp_mode $nlabels $nnoises $c $epochs 2>&1 | tee ../logs/exp"$exp_mode"_d"$data"_g"$gan_type".out