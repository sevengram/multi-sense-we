# -*- coding:utf-8 -*-

import theano
import os

from numpy import random, zeros
from theano import tensor as T
from keras.preprocessing import text, sequence
from six.moves import cPickle

class WordEmbeddingModel(object):
    def __init__(self, words_limit=50000, dimension=128):
        self.words_limit = words_limit
        self.dimension = dimension
        self.tokenizer = text.Tokenizer(nb_words=words_limit)
        self.word_vectors = None

    def build_vocab(self, texts):
        print "Building Vocabulary..."
        self.tokenizer.fit_on_texts(texts)
        self.words_limit = min(self.words_limit, len(self.tokenizer.word_counts))
        self.word_vectors = (random.randn(self.words_limit, self.dimension) - 0.5) / self.dimension
        print "Finish building"

    def _sequentialize(self, texts, **kwargs):
        raise NotImplementedError()

    def fit(self, texts, **kwargs):
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
    def save_model(save_dir):
        print "Save tokenizer..."
        tokenizer_fname = "text8_tokenizer.pkl"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        cPickle.dump(self.tokenizer, open(os.path.join(save_dir, tokenizer_fname), "wb"))

        print "Save word vectors..."
        WV_fname = "text8_WV.pkl"
        cPickle.dump(self.word_vectors, open(os.path.join(save_dir, WV_fname), "wb"))

    def fit(self, texts, lrate=.1, sampling=True, **kwargs):
        x = T.dvector("x")
        y = T.bscalar("y")
        w = theano.shared(zeros((self.words_limit, self.dimension)), name="w")
        b = theano.shared(zeros(self.words_limit), name="b")
        u = T.iscalar("u")

        wu = w[T.cast(u, 'int32')]
        bu = b[T.cast(u, 'int32')]
        hx = 1 / (1 + T.exp(-T.dot(x, wu) - bu))
        gw = (y - hx) * x
        gb = y - hx
        gx = (y - hx) * wu

        train = theano.function(
            inputs=[x, y, u],
            outputs=gx,
            updates=((w, T.inc_subtensor(wu, lrate * gw)), (b, T.inc_subtensor(bu, lrate * gb))))
        print "Start Training"
        count  = 0
        for couples, labels in self._sequentialize(texts, sampling):
            for i, (wi, wj) in enumerate(couples):
                delta_wv = train(self.word_vectors[wi - 1], labels[i], wj - 1)
                self.word_vectors[wi - 1] += lrate * delta_wv
                count += 1
                if count%100 == 0
                    print "Trained", count*100, "couples"
        print "Finish Training"
        save_dir = ""
        print "Save weights"
        weights_fname = "text8_weights.pkl"
        bias_fname = "text8_bias.pkl"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        cPickle.dump(w, open(os.path.join(save_dir, weights_fname), "wb"))
        cPickle.dump(b, open(os.path.join(save_dir, bias_fname), "wb"))
