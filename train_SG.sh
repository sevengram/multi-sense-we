#!/usr/bin/env bash

echo "test 1"
python main.py --model=SG --data=./data/text8 --dimension=128 --window=5 --limit=50000 --min_count=15 --optimizer=sgd \
    --lr=0.001 --objective --save_params --batch=4 --output=/mnt/my_data/we_output --tag=SG_sgd \
    --vocab=/home/ubuntu/multi-sense-we/output/SG_vocab_20151129200134.pkl \

echo "test 2"
python main.py --model=SG --data=./data/text8 --dimension=128 --window=5 --limit=50000 --min_count=15 --optimizer=sgd \
    --lr=0.003 --objective --save_params --batch=4 --output=/mnt/my_data/we_output --tag=SG_sgd \
    --vocab=/home/ubuntu/multi-sense-we/output/SG_vocab_20151129200134.pkl \

echo "test 3"
python main.py --model=SG --data=./data/text8 --dimension=128 --window=5 --limit=50000 --min_count=15 --optimizer=sgd \
    --lr=0.005 --objective --save_params --batch=4 --output=/mnt/my_data/we_output --tag=SG_sgd \
    --vocab=/home/ubuntu/multi-sense-we/output/SG_vocab_20151129200134.pkl \

echo "test 4"
python main.py --model=SG --data=./data/text8 --dimension=128 --window=5 --limit=50000 --min_count=15 --optimizer=sgd \
    --lr=0.01 --objective --save_params --batch=4 --output=/mnt/my_data/we_output --tag=SG_sgd \
    --vocab=/home/ubuntu/multi-sense-we/output/SG_vocab_20151129200134.pkl \

echo "test 5"
python main.py --model=SG --data=./data/text8 --dimension=128 --window=5 --limit=50000 --min_count=15 --optimizer=sgd \
    --lr=0.02 --objective --save_params --batch=4 --output=/mnt/my_data/we_output --tag=SG_sgd \
    --vocab=/home/ubuntu/multi-sense-we/output/SG_vocab_20151129200134.pkl \

echo "test 6"
python main.py --model=SG --data=./data/text8 --dimension=128 --window=5 --limit=50000 --min_count=15 --optimizer=sgd \
    --lr=0.03 --objective --save_params --batch=4 --output=/mnt/my_data/we_output --tag=SG_sgd \
    --vocab=/home/ubuntu/multi-sense-we/output/SG_vocab_20151129200134.pkl \

echo "test 7"
python main.py --model=SG --data=./data/text8 --dimension=128 --window=5 --limit=50000 --min_count=15 --optimizer=sgd \
    --lr=0.04 --objective --save_params --batch=4 --output=/mnt/my_data/we_output --tag=SG_sgd \
    --vocab=/home/ubuntu/multi-sense-we/output/SG_vocab_20151129200134.pkl \

echo "test 8"
python main.py --model=SG --data=./data/text8 --dimension=128 --window=5 --limit=50000 --min_count=15 --optimizer=sgd \
    --lr=0.05 --objective --save_params --batch=4 --output=/mnt/my_data/we_output --tag=SG_sgd \
    --vocab=/home/ubuntu/multi-sense-we/output/SG_vocab_20151129200134.pkl \

echo "test 9"
python main.py --model=SG --data=./data/text8 --dimension=128 --window=5 --limit=50000 --min_count=15 --optimizer=sgd \
    --lr=0.06 --objective --save_params --batch=4 --output=/mnt/my_data/we_output --tag=SG_sgd \
    --vocab=/home/ubuntu/multi-sense-we/output/SG_vocab_20151129200134.pkl \

echo "test 10"
python main.py --model=SG --data=./data/text8 --dimension=128 --window=5 --limit=50000 --min_count=15 --optimizer=sgd \
    --lr=0.07 --objective --save_params --batch=4 --output=/mnt/my_data/we_output --tag=SG_sgd \
    --vocab=/home/ubuntu/multi-sense-we/output/SG_vocab_20151129200134.pkl \

echo "test 11"
python main.py --model=SG --data=./data/text8 --dimension=128 --window=5 --limit=50000 --min_count=15 --optimizer=sgd \
    --lr=0.08 --objective --save_params --batch=4 --output=/mnt/my_data/we_output --tag=SG_sgd \
    --vocab=/home/ubuntu/multi-sense-we/output/SG_vocab_20151129200134.pkl \

echo "test 12"
python main.py --model=SG --data=./data/text8 --dimension=128 --window=5 --limit=50000 --min_count=15 --optimizer=sgd \
    --lr=0.09 --objective --save_params --batch=4 --output=/mnt/my_data/we_output --tag=SG_sgd \
    --vocab=/home/ubuntu/multi-sense-we/output/SG_vocab_20151129200134.pkl \

echo "test 13"
python main.py --model=SG --data=./data/text8 --dimension=128 --window=5 --limit=50000 --min_count=15 --optimizer=sgd \
    --lr=0.1 --objective --save_params --batch=4 --output=/mnt/my_data/we_output --tag=SG_sgd \
    --vocab=/home/ubuntu/multi-sense-we/output/SG_vocab_20151129200134.pkl \

echo "test 14"
python main.py --model=SG --data=./data/text8 --dimension=128 --window=5 --limit=50000 --min_count=15 --optimizer=sgd \
    --lr=0.15 --objective --save_params --batch=4 --output=/mnt/my_data/we_output --tag=SG_sgd \
    --vocab=/home/ubuntu/multi-sense-we/output/SG_vocab_20151129200134.pkl \

echo "test 15"
python main.py --model=SG --data=./data/text8 --dimension=128 --window=5 --limit=50000 --min_count=15 --optimizer=sgd \
    --lr=0.2 --objective --save_params --batch=4 --output=/mnt/my_data/we_output --tag=SG_sgd \
    --vocab=/home/ubuntu/multi-sense-we/output/SG_vocab_20151129200134.pkl \

