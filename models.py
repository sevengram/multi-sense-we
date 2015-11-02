# -*- coding:utf-8 -*-

import cPickle

import theano
from numpy import random, zeros
from theano import tensor as T
from keras.preprocessing import text, sequence


class WordEmbeddingModel(object):
    def __init__(self, words_limit=50000, dimension=128):
        self.words_limit = words_limit
        self.dimension = dimension
        self.tokenizer = None
        self.word_vectors = None

    def init_word_vectors(self):
        self.word_vectors = (random.randn(self.words_limit, self.dimension) - 0.5) / self.dimension

    def build_vocab(self, texts, spare_size=1):
        self.tokenizer = text.Tokenizer(nb_words=self.words_limit)
        self.tokenizer.fit_on_texts(texts)
        self.words_limit = min(self.words_limit, len(self.tokenizer.word_counts)) + spare_size
        self.init_word_vectors()

    def load_vocab(self, path, spare_size=1):
        self.tokenizer = cPickle.load(open(path, 'rb'))
        self.words_limit = min(self.words_limit, len(self.tokenizer.word_counts)) + spare_size
        self.init_word_vectors()

    def save_tokenizer(self, path):
        if path:
            cPickle.dump(self.tokenizer, open(path, "wb"))

    def save_word_vectors(self, path):
        if path:
            cPickle.dump(self.word_vectors, open(path, "wb"))

    def fit(self, texts, **kwargs):
        raise NotImplementedError()

    def _sequentialize(self, texts, **kwargs):
        raise NotImplementedError()


class SkipGramNegSampEmbeddingModel(WordEmbeddingModel):
    def __init__(self, words_limit=50000, dimension=128, window_size=4, neg_sample_rate=1.):
        super(SkipGramNegSampEmbeddingModel, self).__init__(words_limit, dimension)
        self.neg_sample_rate = neg_sample_rate
        self.window_size = window_size

    def _sequentialize(self, texts, sampling=True, **kwargs):
        sampling_table = sequence.make_sampling_table(self.words_limit) if sampling else None
        for seq in self.tokenizer.texts_to_sequences_generator(texts):
            yield sequence.skipgrams(seq, self.words_limit, window_size=self.window_size,
                                     negative_samples=self.neg_sample_rate,
                                     sampling_table=sampling_table)

    def fit(self, texts, lrate=.1, sampling=True, **kwargs):
        x = theano.shared(self.word_vectors, name="x")
        y = T.bscalar("y")
        w = theano.shared(zeros((self.words_limit, self.dimension)), name="w")
        b = theano.shared(zeros(self.words_limit), name="b")
        i = T.iscalar("i")
        j = T.iscalar("j")

        xi = x[T.cast(i, 'int32')]
        wj = w[T.cast(j, 'int32')]
        bj = b[T.cast(j, 'int32')]
        hx = 1 / (1 + T.exp(-T.dot(xi, wj) - bj))
        gx = (y - hx) * wj
        gw = (y - hx) * xi
        gb = y - hx

        train = theano.function(
            inputs=[i, y, j],
            updates=((w, T.inc_subtensor(wj, lrate * gw)),
                     (b, T.inc_subtensor(bj, lrate * gb)),
                     (x, T.inc_subtensor(xi, lrate * gx))))
        for couples, labels in self._sequentialize(texts, sampling):
            for k, (ii, jj) in enumerate(couples):
                train(ii - 1, labels[k], jj - 1)
        self.word_vectors = x.eval()
