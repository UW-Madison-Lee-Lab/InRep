#!/bin/bash
# using ./ or bash

nlabels=50
eval_modes=(4 2)

# data
data=$1
nclass=$2
gtype=$3
exp_mode=$4
ugan=$5
cgan=$6
eval=$7

if [[ $exp_mode -eq 1 ]]
then
  sample_sizes=(10 50 500 1000)
else
  sample_sizes=(15 25 35 50)
fi


#ugan
if [[ $gtype -eq 1 ]] && [[ $ugan -eq 1 ]]
then
  echo 'ugan'
  epochs=100
  bash scripts/ugan.sh $data $nclass $epochs
fi

# training
if [[ $cgan -eq 1 ]]
then
  start=$((0))
  if [[ $gan_type -eq 1 ]]
  then
    end=$((nclass-1))
  else
    end=$((0))
  fi
  for s in ${sample_sizes[@]}
  do
    echo "cgan"
    epochs=50
    bash scripts/cgan.sh $data $nclass $gtype $exp_mode $nlabels $s $epochs $start $end
  done
fi


# eval
if [[ $eval -eq 1 ]]
then
  for e in ${eval_modes[@]}
  do
    for s in ${sample_sizes[@]}
    do
      echo "eval"
      epochs=50
      bash scripts/eval.sh $data $nclass $gtype $exp_mode $nlabels $s $e $epochs
    done
  done
fi
