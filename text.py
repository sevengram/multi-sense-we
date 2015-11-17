# -*- coding:utf-8 -*-

import string


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
    def __init__(self, words_limit=None, filters=base_filter(), lower=True, split=" "):
        self._word_freq = {}
        self._word_index = {}
        self._word_list = []
        self.filters = filters
        self.split = split
        self.lower = lower
        self.words_limit = words_limit

    def fit_on_texts(self, texts):
        """
            required before using texts_to_sequences or texts_to_matrix
            @param texts: can be a list or a generator (for memory-efficiency)
        """
        for text in texts:
            seq = text_to_word_sequence(text, self.filters, self.lower, self.split)
            for w in seq:
                if w in self._word_freq:
                    self._word_freq[w] += 1
                else:
                    self._word_freq[w] = 1
        wcounts = list(self._word_freq.items())
        wcounts.sort(key=lambda x: x[1], reverse=True)
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
                    if self.words_limit and i >= self.words_limit:
                        pass
                    else:
                        vect.append(i)
            yield vect

    def count(self):
        return len(self._word_freq)

    def effective_words(self):
        return self._word_list[:self.words_limit]
