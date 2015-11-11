# -*- coding:utf-8 -*-

import os
import sys
import time
import datetime
import argparse

from models import SkipGramNegSampEmbeddingModel


def build_monitor(start_time, total_lines):
    def m(index, objval):
        percent = float(index) / total_lines
        total_time = (time.time() - start_time) / percent
        remain_time = start_time + total_time - time.time()
        sys.stdout.write(
            '%.2f%%, estimated remaining time: %d min, objective value: %f\r' % (
                percent * 100, int(remain_time / 60), objval))
        sys.stdout.flush()

    return m


def text_generator(path):
    with open(path) as f:
        for l in f:
            yield l


def build_filepath(dirpath, name):
    return "%s/%s_%s.pkl" % (dirpath, name, datetime.datetime.now().strftime('%Y%m%d%H%M%S'))


def file_lines(path):
    i = 0
    with open(path) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', metavar='FILE', help='data file', type=str, required=False)
    parser.add_argument('--dimension', metavar='N', help='Dimension size', type=int, default=128)
    parser.add_argument('--window', metavar='SIZE', help='Window size', type=int, default=5)
    parser.add_argument('--limit', metavar='LIMIT', help='Words limit', type=int, default=5000)
    parser.add_argument('--vocab', metavar='FILE', help='File to load vocab', type=str, required=False)
    parser.add_argument('--wordvec', metavar='FILE', help='File to load word vectors', type=str, required=False)
    parser.add_argument('--output', metavar='FILE', help='Path to save data', type=str, required=False)
    parser.add_argument('--batch', metavar='SIZE', help='Batch size', type=int, default=8)
    parser.add_argument('--epoch', metavar='COUNT', help='Epoch count', type=int, default=1)
    parser.add_argument('--lrate', metavar='RATE', help='Learning rate', type=float, default=.1)
    parser.add_argument("--test", help="Run a manual test after loading/training", action='store_true')
    args = parser.parse_args()

    if args.output and not os.path.exists(args.output):
        os.makedirs(args.output)

    model = SkipGramNegSampEmbeddingModel(words_limit=args.limit, dimension=args.dimension, window_size=args.window)
    if not args.vocab and not args.data:
        print('invalid vocab input')
    if not args.wordvec and not args.data:
        print('invalid word vector input')

    print('start time: %s' % time.ctime())
    print('words_limit=%d, dimension=%d, window_size=%d, batch_size=%d' % (
        model.words_limit, model.dimension, model.window_size, args.batch))

    print('start loading vocab...')
    if args.vocab:
        model.load_vocab(args.vocab)
    else:
        model.build_vocab(text_generator(args.data))

    if args.output and not args.vocab:
        print('saving vocab...')
        model.save_tokenizer(build_filepath(args.output, 'vocab'))
        model.save_word_list(build_filepath(args.output, 'word_list'))
        model.save_word_index(build_filepath(args.output, 'word_index'))

    if args.wordvec:
        print('start loading model...')
        model.load_word_vectors(args.wordvec)
    else:
        print('start fitting model...')
        model.fit(text_generator(args.data), lrate=args.lrate, sampling=True, batch_size=args.batch,
                  monitor=build_monitor(time.time(), file_lines(args.data)))
    print('\nfinish!')

    if args.output and not args.wordvec:
        print('saving word vectors...')
        model.save_word_vectors(build_filepath(args.output, 'word_vec'))
        model.save_weight_matrix(build_filepath(args.output, 'weights'))

    print('end time: %s' % time.ctime())
    print('you may reload the vocab and model, and add --test to run a manual test.')

    while args.test:
        word = raw_input('input a word (\q to exit): ')
        word = word.strip()
        if not word:
            continue
        if word == '\q':
            break
        print(model.nearest_words(word.lower()))
