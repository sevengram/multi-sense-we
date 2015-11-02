# -*- coding:utf-8 -*-

import sys
import time
import argparse

from models import SkipGramNegSampEmbeddingModel


def text_generator(path, total_lines=0):
    with open(path) as f:
        for i, l in enumerate(f):
            if i % 10 == 0 and total_lines != 0:
                sys.stdout.write('%.3f%% %s\r' % (float(i) / total_lines, time.ctime()))
                sys.stdout.flush()
            yield l


def file_lines(path):
    i = 0
    with open(path) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', metavar='file', help='data file', type=str, required=True)
    parser.add_argument('--vout', metavar='file', help='File to save vocab', type=str, required=False)
    parser.add_argument('--wvout', metavar='file', help='File to save word vectors', type=str, required=False)
    args = parser.parse_args()

    model = SkipGramNegSampEmbeddingModel(dimension=64)
    lc = file_lines(args.data)
    print('start time: %s' % time.ctime())
    print('start loading vocab...')
    model.build_vocab(text_generator(args.data))
    print('start fitting model...')
    model.fit(text_generator(args.data, lc), sampling=True)
    print('\nfinish! saving...')
    if args.vout:
        model.save_tokenizer(args.vout)
    if args.wvout:
        model.save_word_vectors(args.wvout)
    print('end time: %s' % time.ctime())
