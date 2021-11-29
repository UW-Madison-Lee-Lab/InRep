#!/bin/bash

data=$1
nclass=$2
gtype=$3
exp=$4
ugan=$5
cgan=$6
eval=$7

echo $data $nclass $gtype $exp $ugan $cgan $eval
bash scripts/main.sh $data $nclass $gtype $exp $ugan $cgan $eval
bash scripts/run.sh 1 10 3 1 0 1 1 2>&1 | tee ../results/logs/exp1_m10_g3.out
