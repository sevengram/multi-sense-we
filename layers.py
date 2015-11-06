# -*- coding:utf-8 -*-

import theano
import theano.tensor as T

from keras import activations
from keras.layers.core import Layer


class WordContextLayer(Layer):
    def __init__(self, input_dim, proj_dim, wordvec_matrix, weight_matrix, biases, activation='sigmoid', **kwargs):
        super(WordContextLayer, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.proj_dim = proj_dim
        self.activation = activations.get(activation)

        self.input = T.imatrix()
        self.wordvec_matrix = theano.shared(wordvec_matrix)
        self.weight_matrix = theano.shared(weight_matrix)
        self.biases = theano.shared(biases)
        self.params = [self.wordvec_matrix, self.weight_matrix, self.biases]

    @property
    def output_shape(self):
        return self.input_shape[0], 1

    def get_output(self, train=False):
        d = self.get_input(train)
        x = self.wordvec_matrix[d[:, 0]]
        w = self.weight_matrix[d[:, 1]]
        b = self.biases[d[:, 1]]
        dot = T.sum(x * w, axis=1)
        dot = theano.tensor.reshape(dot, (d.shape[0], 1))
        return self.activation(dot + b)

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "input_dim": self.input_dim,
                  "proj_dim": self.proj_dim,
                  "activation": self.activation.__name__}
        base_config = super(WordContextLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
