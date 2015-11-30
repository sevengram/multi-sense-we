#python main.py --model=SG --data=./data/text8 --dimension=128 --window=5 --limit=200000 --min_count=15 --optimizer=sgd \
#--lr=0.1 --objective --output=./output/SG_

#python main.py --model=SG --data=./data/text8 --vocab=/Users/qingma/Project/multi-sense-we/output/SG_vocab_20151129200134.pkl \
#--dimension=128 --window=5 --limit=200000 --min_count=15 --optimizer=adagrad --lr=0.3 --objective --output=./output/SG_adagrad_lr03_

#python main.py --model=SG --data=./data/text8 --dimension=128 --window=5 --limit=50000 --min_count=15 --optimizer=sgd \
#--lr=0.05 --objective --save_params=./output/SG_lr005 --batch=4 \
#--vocab=/Users/qingma/Project/multi-sense-we/output/SG_vocab_20151129200134.pkl

python main.py --model=SG --data=./data/text8 --dimension=128 --window=5 --limit=50000 --min_count=15 --optimizer=sgd \
--lr=0.005 --objective --save_params=./output/SG_lr0005_load_ --batch=4 --output=./output/SG_lr0005_load_ \
--vocab=/Users/qingma/Project/multi-sense-we/output/SG_vocab_20151129200134.pkl \
--load_params=/Users/qingma/Project/multi-sense-we/output/SG_lr005parameters_20151129224601.pkl