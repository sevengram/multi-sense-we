from __future__ import absolute_import
import theano
import theano.tensor as T

from keras import activations, initializations
from keras.layers.core import Layer
from keras.utils.theano_utils import shared_zeros


class WordContextProductBiased(Layer):
    '''
        This layer turns a pair of words (a pivot word + a context word,
        ie. a word from the same context, or a random, out-of-context word),
        indentified by their index in a vocabulary, into two dense reprensentations
        (word representation and context representation).

        Then it returns activation(dot(pivot_embedding, context_embedding)),
        which can be trained to encode the probability
        of finding the context word in the context of the pivot word
        (or reciprocally depending on your training procedure).

        The layer ingests integer tensors of shape:
        (nb_samples, 2)
        and outputs a float tensor of shape
        (nb_samples, 1)

        The 2nd dimension encodes (pivot, context).
        input_dim is the size of the vocabulary.

        For more context, see Mikolov et al.:
            Efficient Estimation of Word reprensentations in Vector Space
            http://arxiv.org/pdf/1301.3781v3.pdf
    '''
    input_ndim = 2

    def __init__(self, input_dim, proj_dim=128,
                 init='uniform', activation='sigmoid', weights=None, **kwargs):

        super(WordContextProductBiased, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.proj_dim = proj_dim
        self.init = initializations.get(init)
        self.activation = activations.get(activation)

        self.input = T.imatrix()
        # two different embeddings for pivot word and its context
        # because p(w|c) != p(c|w)
        self.W_w = self.init((input_dim, proj_dim))
        self.W_c = self.init((input_dim, proj_dim))
        self.bias = shared_zeros((input_dim, 1))

        self.params = [self.W_w, self.W_c, self.bias]

        if weights is not None:
            self.set_weights(weights)

    @property
    def output_shape(self):
        return (self.input_shape[0], 1)

    def get_output(self, train=False):
        X = self.get_input(train)
        w = self.W_w[X[:, 0]]  # nb_samples, proj_dim
        c = self.W_c[X[:, 1]]  # nb_samples, proj_dim
        b = self.bias[X[:, 1]]

        dot = T.sum(w * c, axis=1)
        dot = theano.tensor.reshape(dot, (X.shape[0], 1))
        return self.activation(dot + b)

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "input_dim": self.input_dim,
                  "proj_dim": self.proj_dim,
                  "init": self.init.__name__,
                  "activation": self.activation.__name__}
        base_config = super(WordContextProductBiased, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
