# -*- coding:utf-8 -*-

import sys
import time
import datetime
import argparse

from models import SkipGramNegSampEmbeddingModel


def text_generator(path, total_lines=0):
    start_time = time.time()
    with open(path) as f:
        for i, l in enumerate(f):
            if i % 10 == 0 and i != 0 and total_lines != 0:
                percent = float(i) / total_lines
                end_time = start_time + (time.time() - start_time) / percent
                end_time_str = datetime.datetime.fromtimestamp(int(end_time)).ctime()
                sys.stdout.write('%.2f%%, estimated end time: %s\r' % (percent * 100, end_time_str))
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
    parser.add_argument('--data', metavar='FILE', help='data file', type=str, required=False)
    parser.add_argument('--dimension', metavar='N', help='Dimension size', type=int, default=128)
    parser.add_argument('--window-size', metavar='SIZE', help='Window size', type=int, default=5)
    parser.add_argument('--words-limit', metavar='LIMIT', help='Words limit', type=int, default=5000)
    parser.add_argument('--vocab-out', metavar='FILE', help='File to save vocab', type=str, required=False)
    parser.add_argument('--wordvec-out', metavar='FILE', help='File to save word vectors', type=str, required=False)
    parser.add_argument('--vocab-in', metavar='FILE', help='File to load vocab', type=str, required=False)
    parser.add_argument('--wordvec-in', metavar='FILE', help='File to load word vectors', type=str, required=False)
    parser.add_argument("--test", help="Run a manual test after loading/training", action='store_true')
    args = parser.parse_args()

    model = SkipGramNegSampEmbeddingModel(words_limit=args.words_limit, dimension=args.dimension,
                                          window_size=args.window_size)
    if not args.vocab_in and not args.data:
        print('invalid vocab input')
    if not args.wordvec_in and not args.data:
        print('invalid word vector input')

    print('start time: %s' % time.ctime())
    print('words_limit=%d, dimension=%d, window_size=%d' % (model.words_limit, model.dimension, model.window_size))

    print('start loading vocab...')
    if args.vocab_in:
        model.load_vocab(args.vocab_in)
    else:
        model.build_vocab(text_generator(args.data))

    if args.wordvec_in:
        print('start loading model...')
        model.load_word_vectors(args.wordvec_in)
    else:
        print('start fitting model...')
        lc = file_lines(args.data)
        model.fit(text_generator(args.data, lc), sampling=True)
    print('\nfinish!')

    if args.vocab_out and not args.vocab_in:
        model.save_tokenizer(args.vocab_out)
        print('saving vocab...')
    if args.wordvec_out and not args.wordvec_in:
        model.save_word_vectors(args.wordvec_out)
        print('saving word vectors...')

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
