#!/bin/bash
data=cifar10
test=intrafid
gan=inrep
python main.py -d $data -e imbalance -l 1.0 -g $gan -t fid
python main.py -d $data -e imbalance -g $gan -t intrafid
