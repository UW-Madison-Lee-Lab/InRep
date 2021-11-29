#!/bin/bash

#eval exp1 on repgan
data=$1
nclass=$2
gtype=$3

exp=1
ugan=$5
cgan=$6
eval=$7

echo $data $nclass $gtype $exp $ugan $cgan $eval
bash scripts/main.sh $data $nclass $gtype $exp $ugan $cgan $eval