# -*- coding:utf-8 -*-

import math
import string
from collections import defaultdict


def base_filter():
    f = string.punctuation
    f = f.replace("'", '')
    f += '\t\n'
    return f


def text_to_word_sequence(text, filters=base_filter(), lower=True, split=" "):
    """
    prune: sequence of characters to filter out
    """
    if lower:
        text = text.lower()
    text = text.translate(string.maketrans(filters, split * len(filters)))
    seq = text.split(split)
    return [_f for _f in seq if _f]


class Tokenizer(object):
    def __init__(self):
        self._word_freq = defaultdict(int)
        self._word_index = {}
        self.word_list = []
        self.unigram_table = []

    def fit_on_texts(self, texts, words_limit=None, min_count=0, use_stop_words=False):
        """
            required before using texts_to_sequences or texts_to_matrix
            @param texts: can be a list or a generator (for memory-efficiency)
        """
        for text in texts:
            seq = text_to_word_sequence(text)
            for w in seq:
                self._word_freq[w] += 1
        wcounts = list(self._word_freq.items())
        wcounts.sort(key=lambda x: x[1], reverse=True)
        if min_count > 0:
            wcounts = wcounts[:next((i for i, v in enumerate(wcounts) if v[1] < min_count), None)]
        if words_limit:
            wcounts = wcounts[:words_limit]
        if use_stop_words:
            # TODO: use stop words to filter wcounts
            pass
        self.word_list = [wc[0] for wc in wcounts]
        self._word_index = dict(zip(self.word_list, range(len(self.word_list))))
        self.init_unigram_table()

    def texts_to_sequences_generator(self, texts):
        """
            Transform each text in texts in a sequence of integers.
            Only top "nb_words" most frequent words will be taken into account.
            Only words known by the tokenizer will be taken into account.

            Yields individual sequences.
        """
        for text in texts() if callable(texts) else texts:
            seq = text_to_word_sequence(text)
            vect = []
            for w in seq:
                i = self._word_index.get(w)
                if i is not None:
                    vect.append(i)
            yield vect

    def count(self):
        return len(self.word_list)

    def init_unigram_table(self, power=0.75, table_size=10e6):
        words_pow = [math.pow(self._word_freq[w], power) for w in self.word_list]
        s = sum(words_pow)
        i, d = 0, words_pow[0] / s
        self.unigram_table = []
        for a in xrange(int(table_size)):
            self.unigram_table.append(i)
            if float(a) / table_size > d:
                i += 1
                d += words_pow[i] / s
            if i >= len(words_pow):
                i = len(words_pow) - 1
