# -*- coding:utf-8 -*-

import cPickle

import numpy
from numpy import random, zeros, linalg
from theano import tensor as T

import sequence
import text
from trainers import SGD, AdaGrad


class WordEmbeddingModel(object):
    def __init__(self, words_limit=5000, dimension=128, space_factor=1):
        self.words_limit = words_limit
        self.dimension = dimension
        self.space_factor = space_factor
        self.tokenizer = None
        self.word_matrix_index = {}
        self.wordvec_matrix = None
        self.weight_matrix = None
        self.biases = None

    def _init_values(self):
        factor = self.space_factor
        self.wordvec_matrix = (random.randn(self.words_limit * factor, self.dimension).astype(
            numpy.float32) - 0.5) / self.dimension
        self.weight_matrix = zeros((self.words_limit * factor, self.dimension), dtype=numpy.float32)
        self.biases = zeros(self.words_limit * factor, dtype=numpy.float32)

    def build_vocab(self, texts):
        self.tokenizer = text.Tokenizer(words_limit=self.words_limit)
        self.tokenizer.fit_on_texts(texts)
        self.words_limit = min(self.words_limit, len(self.tokenizer.word_counts))
        self._build_word_matrix_index()

    def load_vocab(self, path):
        self.tokenizer = cPickle.load(open(path, 'rb'))
        self.words_limit = min(self.words_limit, len(self.tokenizer.word_counts))
        self._build_word_matrix_index()

    def _build_word_matrix_index(self):
        for i in range(self.words_limit):
            self.word_matrix_index[self.tokenizer.word_list[i]] = [i]

    def load_word_vectors(self, path):
        self.wordvec_matrix = cPickle.load(open(path, 'rb'))

    def save_tokenizer(self, path):
        if path:
            cPickle.dump(self.tokenizer, open(path, "wb"))

    def save_word_list(self, path):
        if path:
            cPickle.dump(self.tokenizer.word_list, open(path, "wb"))

    def save_word_index(self, path):
        if path:
            cPickle.dump(self.word_matrix_index, open(path, "wb"))

    def save_word_vectors(self, path):
        if path:
            cPickle.dump(self.wordvec_matrix, open(path, "wb"))

    def save_weight_matrix(self, path):
        if path:
            cPickle.dump(self.weight_matrix, open(path, "wb"))

    def fit(self, texts, nb_epoch=1, monitor=None, **kwargs):
        raise NotImplementedError()

    def _sequentialize(self, texts, **kwargs):
        raise NotImplementedError()

    def nearest_words(self, word, limit=20):
        if self.tokenizer is None or self.wordvec_matrix is None:
            print('load vocab and model first!')
            return None
        word_index = self.tokenizer.word_index.get(word)
        if word_index is None or word_index >= self.wordvec_matrix.shape[0]:
            print('can\'t find this word!')
            return None
        else:
            d = [linalg.norm(self.wordvec_matrix[word_index] - v) for v in self.wordvec_matrix]
            nearest_indices = numpy.argpartition(d, limit)[:limit]
            return {self.tokenizer.word_list[i]: d[i] for i in nearest_indices}


class SkipGramNegSampEmbeddingModel(WordEmbeddingModel):
    def __init__(self, words_limit=5000, dimension=128, space_factor=1, window_size=5, neg_sample_rate=1.,
                 batch_size=4):
        super(SkipGramNegSampEmbeddingModel, self).__init__(words_limit, dimension, space_factor)
        self.window_size = window_size
        self.batch_size = batch_size
        self.neg_sample_rate = neg_sample_rate
        self.trainer = None

    def _sequentialize(self, texts, sampling=True, **kwargs):
        sampling_table = sequence.make_sampling_table(self.words_limit) if sampling else None
        for seq in self.tokenizer.texts_to_sequences_generator(texts):
            yield seq, sequence.skipgrams(seq, self.words_limit, window_size=self.window_size,
                                          negative_samples=self.neg_sample_rate,
                                          sampling_table=sampling_table)

    def set_trainer(self, lr=.1, optimizer='sgd', **kwargs):
        x = T.fmatrix("x")
        w = T.fmatrix("w")
        b = T.fvector("b")
        y = T.bvector("y")
        hx = 1 / (1 + T.exp(-T.sum(w * x, axis=1) - b))
        obj = y * T.log(hx) + (1 - y) * T.log(1 - hx)

        if optimizer == 'sgd':
            self.trainer = SGD(lr=lr,
                               lr_b=kwargs.get('lr_b'),
                               momentum=kwargs.get('momentum', 0.0),
                               momentum_b=kwargs.get('momentum_b'))
        elif optimizer == 'adagrad':
            self.trainer = AdaGrad(lr=lr,
                                   lr_b=kwargs.get('lr_b'),
                                   epsilon=kwargs.get('epsilon', 1e-6),
                                   gx_shape=(self.batch_size, self.dimension),
                                   gw_shape=(self.batch_size, self.dimension),
                                   gb_shape=self.batch_size)
        else:
            raise NotImplementedError()
        self.trainer.compile(x, w, b, y, hx, obj)

    def fit(self, texts, nb_epoch=1, monitor=None, sampling=True):
        self._init_values()
        for e in range(nb_epoch):
            for k, (seq, (couples, labels, seq_indices)) in enumerate(self._sequentialize(texts, sampling)):
                if callable(monitor) and k % 20 == 0:
                    c = numpy.array(couples)
                    obj = self.trainer.get_objective_value(self.wordvec_matrix[c[:, 0]],
                                                           self.weight_matrix[c[:, 1]],
                                                           self.biases[c[:, 1]],
                                                           labels)
                    monitor(k, obj)
                n = len(couples)
                for i in range(0, n - self.batch_size, self.batch_size):
                    wi, wj = numpy.array(zip(*couples[i:i + self.batch_size]))
                    self.trainer.update(self.wordvec_matrix,
                                        self.weight_matrix,
                                        self.biases,
                                        labels[i:i + self.batch_size],
                                        wi, wj)


class ClusteringSgNsEmbeddingModel(SkipGramNegSampEmbeddingModel):
    def __init__(self, words_limit=5000, dimension=128, space_factor=4, window_size=5, neg_sample_rate=1.):
        super(ClusteringSgNsEmbeddingModel, self).__init__(words_limit, dimension, space_factor, window_size,
                                                           neg_sample_rate)
        self.cluster_center_matrix = None
        self.word_sampling_count = None

    def _init_values(self):
        super(ClusteringSgNsEmbeddingModel, self)._init_values()
        self.cluster_center_matrix = zeros((self.words_limit * self.space_factor, self.dimension), dtype=numpy.float32)
        self.word_sampling_count = zeros(self.words_limit * self.space_factor, dtype=numpy.float32)

    def fit(self, texts, nb_epoch=1, monitor=None, sampling=True):
        self._init_values()
        batch_size = self.batch_size
        for e in range(nb_epoch):
            # TODO: do clustering from epoch 2
            for k, (seq, (couples, labels, seq_indices)) in enumerate(self._sequentialize(texts, sampling)):
                if callable(monitor) and k % 20 == 0:
                    c = numpy.array(couples)
                    obj = self.trainer.get_objective_value(self.wordvec_matrix[c[:, 0]],
                                                           self.weight_matrix[c[:, 1]],
                                                           self.biases[c[:, 1]],
                                                           labels)
                    monitor(k, obj)
                n = len(couples)
                for i in range(0, n - batch_size, batch_size):
                    # get real meaning
                    wi = [self.get_word_meaning(seq, j) for j in seq_indices[i:i + batch_size]]
                    wj = [c[1] for c in couples[i:i + batch_size]]
                    self.trainer.update(self.wordvec_matrix,
                                        self.weight_matrix,
                                        self.biases,
                                        labels[i:i + batch_size],
                                        wi, wj)
                    # update cluster centers
                    centers = [self.cluster_center(seq, j) for j in seq_indices[i:i + batch_size]]
                    p = self.word_sampling_count[wi][:, numpy.newaxis]
                    t = self.cluster_center_matrix[wi] * p + centers
                    self.word_sampling_count[wi] += 1
                    self.cluster_center_matrix[wi] = t / p

    def get_word_meaning(self, seq, i):
        center = self.cluster_center(seq, i)
        mis = self.word_matrix_index[self.tokenizer.word_list[seq[i]]]
        # TODO: 1. split words meaning here  2. add cos distance
        return mis[numpy.argmin(numpy.linalg.norm(self.cluster_center_matrix[mis] - center))]

    def cluster_center(self, seq, i):
        context_words_indices = numpy.asarray(seq)[max(
            0, i - self.window_size): i + self.window_size]  # FIXME: remove center word
        return numpy.mean(self.weight_matrix[context_words_indices], axis=0)
