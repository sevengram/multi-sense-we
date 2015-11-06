# -*- coding:utf-8 -*-

import cPickle

import numpy
import theano
from numpy import random, zeros, linalg
from theano import tensor as T

from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import generic_utils
from keras.preprocessing import text, sequence

from layers import WordContextLayer


class WordEmbeddingModel(object):
    def __init__(self, words_limit=5000, dimension=128):
        self.words_limit = words_limit
        self.dimension = dimension
        self.tokenizer = None
        self.word_list = None
        self.wordvec_matrix = None
        self.weight_matrix = None
        self.biases = None

    def init_values(self):
        self.wordvec_matrix = (random.randn(self.words_limit, self.dimension) - 0.5) / self.dimension
        self.weight_matrix = zeros((self.words_limit, self.dimension))
        self.biases = zeros((self.words_limit, 1))

    def build_vocab(self, texts, spare_size=1):
        self.tokenizer = text.Tokenizer(nb_words=self.words_limit)
        self.tokenizer.fit_on_texts(texts)
        self.words_limit = min(self.words_limit, len(self.tokenizer.word_counts)) + spare_size
        self._build_word_list()

    def load_vocab(self, path, spare_size=1):
        self.tokenizer = cPickle.load(open(path, 'rb'))
        self.words_limit = min(self.words_limit, len(self.tokenizer.word_counts)) + spare_size
        self._build_word_list()

    def load_word_vectors(self, path):
        self.wordvec_matrix = cPickle.load(open(path, 'rb'))

    def save_tokenizer(self, path):
        if path:
            cPickle.dump(self.tokenizer, open(path, "wb"))

    def save_word_vectors(self, path):
        if path:
            cPickle.dump(self.wordvec_matrix, open(path, "wb"))

    def fit(self, texts, nb_epoch=1, **kwargs):
        raise NotImplementedError()

    def _sequentialize(self, texts, **kwargs):
        raise NotImplementedError()

    def _build_word_list(self):
        self.word_list = {v: k for k, v in self.tokenizer.word_index.iteritems()}

    def nearest_words(self, word, limit=20):
        if self.tokenizer is None or self.wordvec_matrix is None:
            print('load vocab and model first!')
            return None
        word_index = self.tokenizer.word_index.get(word)
        if word_index is None or word_index - 1 >= self.wordvec_matrix.shape[0]:
            print('can\'t find this word!')
            return None
        else:
            d = [linalg.norm(self.wordvec_matrix[word_index - 1] - v) for v in self.wordvec_matrix]
            nearest_indices = numpy.argpartition(d, limit)[:limit]
            return {self.word_list[i + 1]: d[i] for i in nearest_indices}


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

    def fit(self, texts, nb_epoch=1, lrate=.1, sampling=True, **kwargs):
        self.init_values()
        x = T.dvector("x")
        y = T.bscalar("y")
        w = T.dvector("w")
        b = T.dscalar("b")

        hx = 1 / (1 + T.exp(-T.dot(x, w) - b))
        gx = (y - hx) * w
        gw = (y - hx) * x
        gb = y - hx

        train = theano.function(
            inputs=[x, y, w, b],
            outputs=[gx, gw, gb])

        for e in range(nb_epoch):
            for couples, labels in self._sequentialize(texts, sampling):
                for i, (wi, wj) in enumerate(couples):
                    dx, dw, db = train(self.wordvec_matrix[wi - 1],
                                       labels[e],
                                       self.weight_matrix[wj - 1],
                                       self.biases[wj - 1])
                    self.wordvec_matrix[wi - 1] += lrate * dx
                    self.weight_matrix[wj - 1] += lrate * dw
                    self.biases[wj - 1] += lrate * db


class KerasEmbeddingModel(SkipGramNegSampEmbeddingModel):
    def fit(self, texts, nb_epoch=1, lrate=.1, sampling=True, **kwargs):
        self.init_values()
        model = Sequential()
        model.add(
            WordContextLayer(self.words_limit, self.dimension, self.wordvec_matrix, self.weight_matrix, self.biases))
        sgd = SGD(lr=lrate)
        model.compile(loss='mse', optimizer=sgd)

        progbar = generic_utils.Progbar(self.tokenizer.document_count)
        losses = []
        for e in range(nb_epoch):
            for i, (couples, labels) in enumerate(self._sequentialize(texts, sampling)):
                if couples:
                    d = numpy.array(couples, dtype="int32")
                    loss = model.train_on_batch(d, labels)
                    losses.append(loss)
                    if len(losses) % 100 == 0:
                        progbar.update(i, values=[("loss", numpy.mean(losses))])
                        losses = []

        self.wordvec_matrix, self.weight_matrix, self.biases = model.layers[0].get_weights()
