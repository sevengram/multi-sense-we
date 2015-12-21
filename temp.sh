#!/usr/bin/env bash

python main.py --model=SG --data=./data/text8 --dimension=128 --window=5 --limit=3000 --min_count=15 --optimizer=sgd \
    --lr=0.05 --objective --batch=32 --output=./output --tag=small_for_eval --epoch=5