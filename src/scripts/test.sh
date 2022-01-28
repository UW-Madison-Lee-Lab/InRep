#!/bin/bash
data=cifar10
test=intrafid
gan=inrep
l=$1
python main.py -d $data -e complexity -l $l -g $gan -t fid
python main.py -d $data -e complexity -l $l -g $gan -t intrafid