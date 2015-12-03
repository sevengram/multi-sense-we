#!/usr/bin/env bash

python main.py --model=SG --data=./data/text8 --dimension=128 --window=5 --limit=50000 \
--vocab=/Users/qingma/Project/multi-sense-we/output/SG_vocab_20151129200134.pkl \
--load_params=/Users/qingma/Project/multi-sense-we/output/SG_lr0005_load_parameters_20151129233821.pkl --test