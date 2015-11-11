# -*- coding:utf-8 -*-

import cPickle

import numpy
import theano
from numpy import random, zeros, linalg
from theano import tensor as T

import sequence
import text


class WordEmbeddingModel(object):
    def __init__(self, words_limit=5000, dimension=128):
        self.words_limit = words_limit
        self.dimension = dimension
        self.tokenizer = None
        self.wordvec_matrix = None
        self.weight_matrix = None
        self.biases = None

    def init_values(self):
        self.wordvec_matrix = (random.randn(self.words_limit, self.dimension).astype(
            numpy.float32) - 0.5) / self.dimension
        self.weight_matrix = zeros((self.words_limit, self.dimension), dtype=numpy.float32)
        self.biases = zeros(self.words_limit, dtype=numpy.float32)

    def build_vocab(self, texts):
        self.tokenizer = text.Tokenizer(words_limit=self.words_limit)
        self.tokenizer.fit_on_texts(texts)
        self.words_limit = min(self.words_limit, len(self.tokenizer.word_counts))

    def load_vocab(self, path):
        self.tokenizer = cPickle.load(open(path, 'rb'))
        self.words_limit = min(self.words_limit, len(self.tokenizer.word_counts))

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
            cPickle.dump(self.tokenizer.word_index, open(path, "wb"))

    def save_word_vectors(self, path):
        if path:
            cPickle.dump(self.wordvec_matrix, open(path, "wb"))

    def save_weight_matrix(self, path):
        if path:
            cPickle.dump(self.weight_matrix, open(path, "wb"))

    def fit(self, texts, nb_epoch=1, **kwargs):
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
    def __init__(self, words_limit=5000, dimension=128, window_size=5, neg_sample_rate=1.):
        super(SkipGramNegSampEmbeddingModel, self).__init__(words_limit, dimension)
        self.neg_sample_rate = neg_sample_rate
        self.window_size = window_size

    def _sequentialize(self, texts, sampling=True, **kwargs):
        sampling_table = sequence.make_sampling_table(self.words_limit) if sampling else None
        for seq in self.tokenizer.texts_to_sequences_generator(texts):
            yield sequence.skipgrams(seq, self.words_limit, window_size=self.window_size,
                                     negative_samples=self.neg_sample_rate,
                                     sampling_table=sampling_table)

    def fit_graph(self, texts, nb_epoch=1, lrate=.1, sampling=True):
        self.init_values()
        x = T.fvector("x")
        y = T.bscalar("y")
        w = T.fvector("w")
        b = T.fscalar("b")

        hx = 1 / (1 + T.exp(-T.dot(w, x) - b))
        obj = y*T.log(hx) + (1-y)*T.log(1-hx)

        gx, gw, gb = T.grad(obj, [x, w, b])

        train = theano.function(
            inputs=[x, y, w, b],
            outputs=[gx, gw, gb])

        for e in range(nb_epoch):
            for couples, labels in self._sequentialize(texts, sampling):
                for i, (wi, wj) in enumerate(couples):
                    dx, dw, db = train(self.wordvec_matrix[wi],
                                       labels[i],
                                       self.weight_matrix[wj],
                                       self.biases[wj])
                    self.wordvec_matrix[wi] += lrate * dx
                    self.weight_matrix[wj] += lrate * dw
                    self.biases[wj] += lrate * db

    def fit(self, texts, nb_epoch=1, lrate=.1, sampling=True, **kwargs):
        self.init_values()
        x = T.fvector("x")
        y = T.bscalar("y")
        w = T.fvector("w")
        b = T.fscalar("b")

        hx = 1 / (1 + T.exp(-T.dot(w, x) - b))
        gb = y - hx
        gx = gb * w
        gw = gb * x

        train = theano.function(
            inputs=[x, y, w, b],
            outputs=[gx, gw, gb])

        for e in range(nb_epoch):
            for couples, labels in self._sequentialize(texts, sampling):
                for i, (wi, wj) in enumerate(couples):
                    dx, dw, db = train(self.wordvec_matrix[wi],
                                       labels[i],
                                       self.weight_matrix[wj],
                                       self.biases[wj])
                    self.wordvec_matrix[wi] += lrate * dx
                    self.weight_matrix[wj] += lrate * dw
                    self.biases[wj] += lrate * db

    def batch_fit(self, texts, nb_epoch=1, lrate=.1, sampling=True, batch_size=8):
        self.init_values()
        x = T.fmatrix("x")
        y = T.bvector("y")
        w = T.fmatrix("w")
        b = T.fvector("b")

        hx = 1 / (1 + T.exp(-T.sum(w * x, axis=1) - b))
        obj = y*T.log(hx) + (1-y)*T.log(1-hx)
        avg_obj = T.mean(obj)
        gb = y - hx
        gx = T.transpose(gb * T.transpose(w))
        gw = T.transpose(gb * T.transpose(x))

        get_obj = theano.function(inputs=[x, y, w, b], outputs=[avg_obj])

        train = theano.function(
            inputs=[x, y, w, b],
            outputs=[gx, gw, gb])

        for e in range(nb_epoch):
            c = 0
            for couples, labels in self._sequentialize(texts, sampling):
                n = len(couples)
                for i in range(0, n, batch_size):
                    wi, wj = numpy.array(zip(*couples[i:i + batch_size]))
                    dx, dw, db = train(self.wordvec_matrix[wi],
                                       labels[i:i + batch_size],
                                       self.weight_matrix[wj],
                                       self.biases[wj])
                    self.wordvec_matrix[wi] += lrate * dx
                    self.weight_matrix[wj] += lrate * dw
                    self.biases[wj] += lrate * db
                if c % 20 == 0 and c != 0:
                    X = numpy.array(couples, dtype="int32")
                    avg_obj = get_obj(self.wordvec_matrix[X[:, 0]],
                                      labels,
                                      self.weight_matrix[X[:, 1]],
                                      self.biases[X[:, 1]])
                    print "average objective value: ", avg_obj[0]
                c += 1

    def batch_fit_graph(self, texts, nb_epoch=1, lrate=.1, sampling=True, batch_size=5, **kwargs):
        self.init_values()
        x = T.fmatrix("x")
        y = T.bvector("y")
        w = T.fmatrix("w")
        b = T.fvector("b")

        hx = 1 / (1 + T.exp(-T.sum(w * x, axis=1) - b))
        obj = y*T.log(hx) + (1-y)*T.log(1-hx)
        cost = T.sum(obj)
        gx, gw, gb = T.grad(cost, [x, w, b])

        train = theano.function(
            inputs=[x, y, w, b],
            outputs=[gx, gw, gb])

        for e in range(nb_epoch):
            for couples, labels in self._sequentialize(texts, sampling):
                n = len(couples)
                for i in range(0, n, batch_size):
                    wi, wj = numpy.array(zip(*couples[i:i + batch_size]))
                    dx, dw, db = train(self.wordvec_matrix[wi],
                                       labels[i:i + batch_size],
                                       self.weight_matrix[wj],
                                       self.biases[wj])
                    self.wordvec_matrix[wi] += lrate * dx
                    self.weight_matrix[wj] += lrate * dw
                    self.biases[wj] += lrate * db