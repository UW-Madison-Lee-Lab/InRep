#!/bin/bash
data=cifar10
test=intrafid
gan=inrep
python main.py -d $data -e asymnoise -g $gan -t fid
python main.py -d $data -e asymnoise -g $gan -t intrafid