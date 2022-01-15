#!/bin/bash
data=cifar10
test=intrafid
gan=inrep
python main.py -d $data -e complexity -l 0.1 -g $gan -t $test
