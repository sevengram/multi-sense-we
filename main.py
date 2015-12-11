# -*- coding:utf-8 -*-

import os
import sys
import time
import cPickle
import datetime
import argparse

from models import ClusteringSgNsEmbeddingModel, SkipGramNegSampEmbeddingModel, InteractiveClSgNsEmbeddingModel


def build_monitor(total_lines, monitor_values=None):
    start_time = time.time()

    def m(index, objval):
        if monitor_values is not None:
            monitor_values.append(objval)
        if index != 0:
            percent = float(index) / total_lines
            total_time = (time.time() - start_time) / percent
            remain_time = start_time + total_time - time.time()
            sys.stdout.write(
                '%.2f%%, estimated remaining time: %d sec, objective value: %f\r' % (
                    percent * 100, remain_time, objval))
            sys.stdout.flush()

    return m


def text_builder(path):
    def g():
        with open(path) as f:
            for l in f:
                yield l

    return g


def build_filepath(dirpath, tags, name):
    return "%s/%s_%s.pkl" % (dirpath, tags, name)


def build_sub_dirpath(dirpath, tags):
    return "%s/%s_%s" % (dirpath, tags, datetime.datetime.now().strftime('%m%d%H%M%S'))


def file_lines(path):
    i = 0
    with open(path) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', metavar='MODEL', help='which model to use', type=str, default='CL')
    parser.add_argument('--data', metavar='FILE', help='data file', type=str, required=False)
    parser.add_argument('--dimension', metavar='N', help='Dimension size', type=int, default=128)
    parser.add_argument('--window', metavar='SIZE', help='Window size', type=int, default=5)
    parser.add_argument('--neg', metavar='LIMIT', help='negative sample rate', type=int, default=1)
    parser.add_argument('--limit', metavar='LIMIT', help='Words limit', type=int, default=40000)
    parser.add_argument('--min_count', metavar='COUNT', help='Minimum count using', type=int, required=False)
    parser.add_argument('--vocab', metavar='FILE', help='File to load vocab', type=str, required=False)
    parser.add_argument('--wordvec', metavar='FILE', help='File to load word vectors', type=str, required=False)
    parser.add_argument('--output', metavar='FILE', help='Path to save data', type=str, required=False)
    parser.add_argument('--tag', metavar='FILE', help='Tags to give key parameters', type=str, default='')
    parser.add_argument('--save_params', help='Whether save parameters', action='store_true')
    parser.add_argument('--load_params', metavar='FILE', help='Path to load parameters', type=str, required=False)
    parser.add_argument('--batch', metavar='SIZE', help='Batch size', type=int, default=8)
    parser.add_argument('--learnMultiTop', metavar='SIZE', help='Only learn multiple vectors for top V words',
                        type=int, default=4000)
    parser.add_argument('--skiplist', metavar='FILE', help='Use skip list to skip single sense words', type=str,
                        required=False)
    parser.add_argument('--epoch', metavar='COUNT', help='Epoch count', type=int, default=1)
    parser.add_argument('--lr', metavar='RATE', help='Learning rate', type=float, default=.1)
    parser.add_argument('--lr_b', metavar='RATE', help='Learning rate for bias', type=float, required=False)
    parser.add_argument('--momentum', metavar='RATE', help='Momentum', type=float, default=.0)
    parser.add_argument('--momentum_b', metavar='RATE', help='Momentum for bias', type=float, required=False)
    parser.add_argument('--optimizer', metavar='TYPE', help='Optimizer type', type=str, default='sgd')
    parser.add_argument('--objective', help='Save objective value or not', action='store_true')
    parser.add_argument("--snapshot", help="Take snapshot while training", action='store_true')
    parser.add_argument('--save_vec', help='Save word vectors and context vectors', action='store_true')
    parser.add_argument("--test", help="Run a manual test after loading/training", action='store_true')
    args = parser.parse_args()

    if args.output and not os.path.exists(args.output):
        os.makedirs(args.output)

    sub_dir = ""
    if args.output:
        sub_dir = build_sub_dirpath(args.output, args.tag)
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)
        else:
            sub_dir += '_' + datetime.datetime.now().strftime('%S')
            os.makedirs(sub_dir)
        f = open(sub_dir + "/arguments.txt", 'w')
        f.write("         data: " + str(args.data) + "\n")
        f.write("        vocab: " + str(args.vocab) + "\n")
        f.write("        limit: " + str(args.limit) + "\n")
        f.write("    min_count: " + str(args.min_count) + "\n")
        f.write("       window: " + str(args.window) + "\n")
        f.write("          neg: " + str(args.neg) + "\n")
        f.write("        model: " + str(args.model) + "\n")
        f.write("  load_params: " + str(args.load_params) + "\n")
        f.write("        epoch: " + str(args.epoch) + "\n")
        f.write("        batch: " + str(args.batch) + "\n")
        f.write("           lr: " + str(args.lr) + "\n")
        f.write("         lr_b: " + str(args.lr_b) + "\n")
        f.write("     momentum: " + str(args.momentum) + "\n")
        f.write("   momentum_b: " + str(args.momentum_b) + "\n")
        f.write("    optimizer: " + str(args.optimizer) + "\n")
        f.write("    dimension: " + str(args.dimension) + "\n")
        f.write("learnMultiTop: " + str(args.learnMultiTop) + "\n")
        f.write("     skiplist: " + str(args.skiplist) + "\n")
        f.write("      wordvec: " + str(args.wordvec) + "\n")
        f.close()

    multi_sense_skip_list = cPickle.load(open(args.skiplist, 'rb')) if args.skiplist else None

    model = None
    if args.model == 'CL':
        model = ClusteringSgNsEmbeddingModel(words_limit=args.limit, dimension=args.dimension, window_size=args.window,
                                             batch_size=args.batch, min_count=args.min_count, neg_sample_rate=args.neg)
    elif args.model == 'SG':
        model = SkipGramNegSampEmbeddingModel(words_limit=args.limit, dimension=args.dimension, window_size=args.window,
                                              batch_size=args.batch, min_count=args.min_count, neg_sample_rate=args.neg)
    elif args.model == 'IC':
        model = InteractiveClSgNsEmbeddingModel(words_limit=args.limit, dimension=args.dimension,
                                                window_size=args.window, batch_size=args.batch,
                                                min_count=args.min_count, neg_sample_rate=args.neg)
    else:
        NotImplementedError()

    if not args.vocab and not args.data:
        print('invalid vocab input')
    if not args.wordvec and not args.data:
        print('invalid word vector input')

    print('start time: %s' % time.ctime())
    print('words_limit=%d, dimension=%d, window_size=%d, batch_size=%d' % (
        model.words_limit, model.dimension, model.window_size, args.batch))

    print('start loading vocab...')
    if args.vocab:
        model.load_vocab(args.vocab, multi_sense_skip_list, args.learnMultiTop)
    else:
        model.build_vocab(text_builder(args.data)(), multi_sense_skip_list, args.learnMultiTop)

    if args.output and not args.vocab:
        print('saving vocab...')
        model.save_tokenizer(build_filepath(sub_dir, args.tag, 'vocab'))
        model.save_word_list(build_filepath(sub_dir, args.tag, 'word_list'))
        model.save_word_index(build_filepath(sub_dir, args.tag, 'word_index'))

    obj_trajectory = [] if args.objective else None

    if args.wordvec:
        print('start loading word vectors...')
        model.load_word_vectors(args.wordvec)

    if args.load_params:
        print('loading previous parameters...')
        model.load(args.load_params)

    if not args.test:
        print('start fitting model...')
        snapshot_path_base = None
        if args.snapshot:
            params_path = build_filepath(sub_dir, args.tag, 'params')
            snapshot_path_base = params_path[:-4]
        model.set_trainer(lr=args.lr, lr_b=args.lr_b, momentum=args.momentum, momentum_b=args.momentum_b,
                          optimizer=args.optimizer)
        model.fit(text_builder(args.data), nb_epoch=args.epoch,
                  monitor=build_monitor(file_lines(args.data), obj_trajectory),
                  snapshot_path=snapshot_path_base if args.snapshot else None)
        print('\nfinish!')

    if args.save_params and not args.snapshot:
        print('saveing all parameters...')
        model.dump(build_filepath(sub_dir, args.tag, 'params'))

    if args.output and args.save_vec and not args.save_params:
        print('saving word vectors...')
        model.save_word_vectors(build_filepath(sub_dir, args.tag, 'word_vec'))
        model.save_weight_matrix(build_filepath(sub_dir, args.tag, 'weights'))

    if args.output and args.objective:
        cPickle.dump(obj_trajectory, open(build_filepath(sub_dir, args.tag, 'objective'), "wb"))
        f = open(sub_dir + '/' + args.tag + '_objective.txt', 'w')
        for obj in obj_trajectory:
            f.write(str(obj) + '\n')
        f.close()

    print('end time: %s' % time.ctime())
    print('you may reload the vocab and model, and add --test to run a manual test.')

    while args.test:
        word = raw_input('input a word (\q to exit): ')
        word = word.strip()
        if not word:
            continue
        if word == '\q':
            break
        if args.model == 'SG':
            print(model.nearest_words(word.lower()))
        elif args.model == 'CL':
            word_list = model.nearest_words(word.lower())
            for i, nearest_sense in enumerate(word_list):
                print "sense", i, ":"
                print(nearest_sense)
        else:
            NotImplementedError()
