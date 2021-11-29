#!/bin/bash
# gan=minegan
# for l in 0.1 0.2
# do 
#     python main.py -d cifar10 -e complexity -l $l -g $gan -t $test
# done
# echo $gan

data=cifar10
test=visual

# python main.py -d cifar10 -e complexity -l 0.2 -g repgan --is_train -t intrafid -c 0
# contragan, 1.0 -- networks
# ganrep 0.01, 1.0
# repgan 0.2 0.5

for gan in projgan
do
    for l in 0.01 0.1 0.2 0.5 1.0
    do 
        python main.py -d $data -e complexity -l $l -g $gan -t $test 
    done
done

# sh scripts/test.sh cifar10 intrafid acgan 2>&1 | tee ../logs/cifar10_acgan_intra.txt