# -*- coding:utf-8 -*-

import os
import sys
import time
import cPickle
import datetime
import argparse

from models import ClusteringSgNsEmbeddingModel


def build_monitor(total_lines, monitor_values=None):
    start_time = time.time()

    def m(index, objval):
        if monitor_values:
            monitor_values.append(objval)
        percent = (float(index) + 1.0) / total_lines
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
    parser.add_argument('--lr', metavar='RATE', help='Learning rate', type=float, default=.1)
    parser.add_argument('--lr_b', metavar='RATE', help='Learning rate for bias', type=float, required=False)
    parser.add_argument('--momentum', metavar='RATE', help='Momentum', type=float, default=.0)
    parser.add_argument('--momentum_b', metavar='RATE', help='Momentum for bias', type=float, required=False)
    parser.add_argument('--optimizer', metavar='TYPE', help='Optimizer type', type=str, default='sgd')
    parser.add_argument('--objective', help='Save objective value or not', action='store_true')
    parser.add_argument("--test", help="Run a manual test after loading/training", action='store_true')
    args = parser.parse_args()

    if args.output and not os.path.exists(args.output):
        os.makedirs(args.output)

    model = ClusteringSgNsEmbeddingModel(words_limit=args.limit, dimension=args.dimension, window_size=args.window)
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

    if args.objective:
        obj_trajectory = []
    else:
        obj_trajectory = None

    if args.wordvec:
        print('start loading model...')
        model.load_word_vectors(args.wordvec)
    else:
        print('start fitting model...')
        model.set_trainer(optimizer=args.optimizer)
        model.fit(text_generator(args.data), monitor=build_monitor(file_lines(args.data), obj_trajectory))
    print('\nfinish!')

    if args.output and not args.wordvec:
        print('saving word vectors...')
        model.save_word_vectors(build_filepath(args.output, 'word_vec'))
        model.save_weight_matrix(build_filepath(args.output, 'weights'))

    if args.output and args.objective:
        cPickle.dump(obj_trajectory, open(build_filepath(args.output, 'objective'), "wb"))

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
