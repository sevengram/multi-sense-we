# -*- coding:utf-8 -*-

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
    def __init__(self, words_limit=None, min_count=None, filters=base_filter(), lower=True, split=" ",
                 use_stop_words=False):
        self._word_freq = defaultdict(int)
        self._word_index = {}
        self._word_list = []
        self.filters = filters
        self.split = split
        self.lower = lower
        self.words_limit = words_limit
        self.min_count = min_count
        self.use_stop_words = use_stop_words

    def fit_on_texts(self, texts):
        """
            required before using texts_to_sequences or texts_to_matrix
            @param texts: can be a list or a generator (for memory-efficiency)
        """
        for text in texts:
            seq = text_to_word_sequence(text, self.filters, self.lower, self.split)
            for w in seq:
                self._word_freq[w] += 1
        wcounts = list(self._word_freq.items())
        wcounts.sort(key=lambda x: x[1], reverse=True)

        wsize = len(wcounts)
        limits = 0
        if self.min_count:
            for i in xrange(wsize):
                if wcounts[wsize-i-1][1] >= self.min_count:
                    limits = i
                    break
            wcounts = wcounts[:limits]

        if self.use_stop_words:
            # TODO: use stop words to filter wcounts
            pass

        if self.words_limit:
            wcounts = wcounts[:self.words_limit]

        self._word_list = [wc[0] for wc in wcounts]
        self._word_index = dict(zip(self._word_list, range(len(self._word_list))))

    def texts_to_sequences_generator(self, texts):
        """
            Transform each text in texts in a sequence of integers.
            Only top "nb_words" most frequent words will be taken into account.
            Only words known by the tokenizer will be taken into account.

            Yields individual sequences.
        """
        for text in texts:
            seq = text_to_word_sequence(text, self.filters, self.lower, self.split)
            vect = []
            for w in seq:
                i = self._word_index.get(w)
                if i is not None:
                    vect.append(i)
            yield vect

    def provide_word_list(self):
        return self._word_list

    def count(self):
        return len(self._word_list)
