#!/bin/bash

#!/bin/bash
# gan=minegan
# for l in 0.1 0.2
# do 
#     python main.py -d cifar10 -e complexity -l $l -g $gan -t $test
# done
# echo $gan

# data=cifar10
# test=fid
# for gan in contragan ganrep minegan repgan
# do
#     echo $gan
#     for l in 0.01
#     do 
#         python main.py -d $data -e complexity -l $l -g $gan -t $test 
#     done
# done
# sh scripts/test.sh cifar10 intrafid acgan 2>&1 | tee ../logs/cifar10_acgan_intra.txt

for c in $(seq 0 9)
do
    python main.py -d cifar10 -e complexity -l $1 -g ganrep -c $c --is_train -n 10
done
python main.py -d cifar10 -e complexity -l $1 -g ganrep -t visual