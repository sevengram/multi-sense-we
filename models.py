# -*- coding:utf-8 -*-

import cPickle

import numpy
import theano
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
    def __init__(self, words_limit=5000, dimension=128, space_factor=1, window_size=5, neg_sample_rate=1.):
        super(SkipGramNegSampEmbeddingModel, self).__init__(words_limit, dimension, space_factor)
        self.window_size = window_size
        self.neg_sample_rate = neg_sample_rate
        self.word_sampling_count = None

    def _init_values(self):
        super(SkipGramNegSampEmbeddingModel, self)._init_values()
        self.word_sampling_count = zeros(self.words_limit * self.space_factor, dtype=numpy.float32)

    def _sequentialize(self, texts, sampling=True, **kwargs):
        sampling_table = sequence.make_sampling_table(self.words_limit) if sampling else None
        for seq in self.tokenizer.texts_to_sequences_generator(texts):
            yield seq, sequence.skipgrams(seq, self.words_limit, window_size=self.window_size,
                                          negative_samples=self.neg_sample_rate,
                                          sampling_table=sampling_table)

    def fit(self, texts, nb_epoch=1, monitor=None, lrate=.1, sampling=True, batch_size=8, **kwargs):
        self._init_values()
        x = T.fmatrix("x")
        y = T.bvector("y")
        w = T.fmatrix("w")
        b = T.fvector("b")

        hx = 1 / (1 + T.exp(-T.sum(w * x, axis=1) - b))
        gb = y - hx
        gx = T.transpose(gb * T.transpose(w))
        gw = T.transpose(gb * T.transpose(x))
        gradient = theano.function(
            inputs=[x, y, w, b],
            outputs=[gx, gw, gb])

        obj = y * T.log(hx) + (1 - y) * T.log(1 - hx)
        objval = theano.function(
            inputs=[x, y, w, b],
            outputs=T.mean(obj))

        for e in range(nb_epoch):
            for k, (seq, (couples, labels, seq_indices)) in enumerate(self._sequentialize(texts, sampling)):
                n = len(couples)
                for i in range(0, n, batch_size):
                    wi, wj = numpy.array(zip(*couples[i:i + batch_size]))
                    dx, dw, db = gradient(self.wordvec_matrix[wi],
                                          labels[i:i + batch_size],
                                          self.weight_matrix[wj],
                                          self.biases[wj])
                    self.wordvec_matrix[wi] += lrate * dx
                    self.weight_matrix[wj] += lrate * dw
                    self.biases[wj] += lrate * db
                if callable(monitor) and k % 20 == 0 and k != 0:
                    c = numpy.array(couples)
                    obj = objval(self.wordvec_matrix[c[:, 0]],
                                 labels,
                                 self.weight_matrix[c[:, 1]],
                                 self.biases[c[:, 1]])
                    monitor(k, obj)

    def fit_bis(self, texts, nb_epoch=1, monitor=None, sampling=True, lr=.1, momentum=0.0, batch_size=4, epsilon=1e-6,
                optimizer='sgd', **kwargs):
        self._init_values()
        x = T.fmatrix("x")
        y = T.bvector("y")
        w = T.fmatrix("w")
        b = T.fvector("b")

        hx = 1 / (1 + T.exp(-T.sum(w * x, axis=1) - b))
        obj = y * T.log(hx) + (1 - y) * T.log(1 - hx)
        obj_mean = T.mean(obj)
        objval = theano.function(
            inputs=[x, y, w, b],
            outputs=obj_mean)

        if optimizer == 'sgd':
            trainer = SGD(lr=lr, momentum=momentum)
        elif optimizer == 'adagrad':
            trainer = AdaGrad(lr=lr, epsilon=epsilon)
        else:
            raise NotImplementedError()
        gradient = trainer.compile(x, w, b, y, obj_mean)

        for e in range(nb_epoch):
            for k, (seq, (couples, labels, seq_indices)) in enumerate(self._sequentialize(texts, sampling)):
                n = len(couples)
                for i in range(0, n, batch_size):
                    wi, wj = numpy.array(zip(*couples[i:i + batch_size]))
                    trainer.update(wi, wj, self.wordvec_matrix, labels[i:i + batch_size],
                                   self.weight_matrix, self.biases, gradient)
                if callable(monitor) and k % 20 == 0 and k != 0:
                    c = numpy.array(couples)
                    obj = objval(self.wordvec_matrix[c[:, 0]],
                                 labels,
                                 self.weight_matrix[c[:, 1]],
                                 self.biases[c[:, 1]])
                    monitor(k, obj)


class ClusteringSgNsEmbeddingModel(SkipGramNegSampEmbeddingModel):
    def __init__(self, words_limit=5000, dimension=128, space_factor=4, window_size=5, neg_sample_rate=1.):
        super(ClusteringSgNsEmbeddingModel, self).__init__(words_limit, dimension, space_factor, window_size,
                                                           neg_sample_rate)
        self.cluster_center_matrix = None

    def _init_values(self):
        super(ClusteringSgNsEmbeddingModel, self)._init_values()
        self.cluster_center_matrix = zeros((self.words_limit * self.space_factor, self.dimension), dtype=numpy.float32)

    def clustering(self, seq, seq_indices):
        # TODO
        pass

    def cluster_center(self, context_words_indices):
        return numpy.mean(self.weight_matrix[context_words_indices], axis=0)
