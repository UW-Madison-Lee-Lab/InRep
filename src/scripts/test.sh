#!/bin/bash
data=cifar10
test=intrafid
gan=inrep
# for l in 0.01 0.1 0.2 0.5 1.0
# do
#     python main.py -d $data -e complexity -l $l -g $gan -t fid
#     python main.py -d $data -e complexity -l $l -g $gan -t intrafid
# done
l=$1
python main.py -d $data -e complexity -l $l -g $gan -t fid
python main.py -d $data -e complexity -l $l -g $gan -t intrafid