import theano
import numpy
from theano import tensor as T
from numpy import zeros, ones


class Trainer(object):
    def __init__(self, lr=.1, lr_b=None):
        self.lr = lr
        if lr_b is not None:
            self.lr_b = lr_b
        else:
            self.lr_b = lr

    def compile(self, x, w, b, y, obj_func):
        raise NotImplementedError

    def update(self, wi, wj, embeds, labels, weights, bias, gradients):
        raise NotImplementedError


class SGD(Trainer):
    def __init__(self, lr=.1, momentum=0.0, lr_b=None, momentum_b=None ):
        super(SGD, self).__init__(lr, lr_b)
        if momentum_b is not None:
            self.momentum_b = momentum_b
        else:
            self.momentum_b = momentum
        self.momentum = momentum

    def compile(self, x, w, b, y, hx):
        gb = y - hx
        gx = T.transpose(gb * T.transpose(w))
        gw = T.transpose(gb * T.transpose(x))

        gradients = theano.function(
            inputs=[x, y, w, b],
            outputs=[gx, gw, gb])
        return gradients

    def update(self, wi, wj, embeds, labels, weights, bias, gradients):
        dx, dw, db = gradients(embeds[wi], labels, weights[wj], bias[wj])
        update_x = (1.0 - self.momentum) * self.lr * dx - self.momentum * embeds[wi]
        update_w = (1.0 - self.momentum) * self.lr * dw - self.momentum * weights[wj]
        update_b = (1.0 - self.momentum_b) * self.lr_b * db - self.momentum_b * bias[wj]

        embeds[wi] += update_x
        weights[wj] += update_w
        bias[wj] += update_b


class AdaGrad(Trainer):
    def __init__(self, lr=.1, lr_b=None, epsilon=1e-5, gx_shape=None, gw_shape=None, gb_shape=None):
        super(AdaGrad, self).__init__(lr, lr_b)
        self.epsilon = numpy.array(epsilon, dtype=numpy.float32)
        self.acc_gx = zeros(gx_shape, dtype=numpy.float32)
        self.acc_gw = zeros(gw_shape, dtype=numpy.float32)
        self.acc_gb = zeros(gb_shape, dtype=numpy.float32)

    def compile(self, x, w, b, y, hx):
        acc_x = T.fmatrix('acc_x')
        acc_w = T.fmatrix('acc_w')
        acc_b = T.fvector('acc_b')

        gb = y - hx
        gx = T.transpose(gb * T.transpose(w))
        gw = T.transpose(gb * T.transpose(x))

        acc_nx = acc_x + gx**2
        acc_nw = acc_w + gw**2
        acc_nb = acc_b + gb**2

        gx_new = gx/T.sqrt(acc_x + self.epsilon)
        gw_new = gw/T.sqrt(acc_w + self.epsilon)
        gb_new = gb/T.sqrt(acc_b + self.epsilon)

        gradients = theano.function(inputs=[acc_x, acc_w, acc_b, x, y, w, b], outputs=[gx_new, gw_new, gb_new])
        accumulator = theano.function(inputs=[acc_x, acc_w, acc_b, x, y, w, b], outputs=[acc_nx, acc_nw, acc_nb])

        return [gradients, accumulator]

    def update(self, wi, wj, embeds, labels, weights, bias, update_funcs):
        gradients = update_funcs[0]
        accumulator = update_funcs[1]

        self.acc_gx, self.acc_gw, self.acc_gb = accumulator(self.acc_gx, self.acc_gw, self.acc_gb,
                                                            embeds[wi], labels, weights[wj], bias[wj])
        dx, dw, db = gradients(self.acc_gx, self.acc_gw, self.acc_gb,
                               embeds[wi], labels, weights[wj], bias[wj])

        embeds[wi] += self.lr * dx
        weights[wj] += self.lr * dw
        bias[wj] += self.lr_b * db