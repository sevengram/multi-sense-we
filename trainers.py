import theano
import numpy
from theano import tensor as T
from numpy import zeros


class Trainer(object):
    def __init__(self, lr=.1, lr_b=None):
        self.lr = lr
        if lr_b is None:
            self.lr_b = lr
        else:
            self.lr_b = lr_b
        self.gradient = None
        self.objective = None

    def compile(self, x, w, b, y, obj):
        raise NotImplementedError

    def update(self, embeds, weights, bias, labels, wi, wj):
        raise NotImplementedError

    def objective_value(self, embeds, weights, bias, labels):
        return self.objective(embeds, weights, bias, labels)


class SGD(Trainer):
    def __init__(self, lr=.1, lr_b=None, momentum=0.0, momentum_b=None):
        super(SGD, self).__init__(lr, lr_b)
        self.momentum = momentum
        self.momentum_b = momentum_b or momentum

    def compile(self, x, w, b, y, obj):
        obj_mean = T.mean(obj)
        obj_sum = T.sum(obj)
        gx, gw, gb = T.grad(obj_sum, [x, w, b])
        self.gradient = theano.function(
            inputs=[x, w, b, y],
            outputs=[gx, gw, gb])
        self.objective = theano.function(
            inputs=[x, w, b, y],
            outputs=obj_mean)

    def update(self, embeds, weights, bias, labels, wi, wj):
        dx, dw, db = self.gradient(embeds[wi], weights[wj], bias[wj], labels)
        update_x = (1.0 - self.momentum) * self.lr * dx - self.momentum * embeds[wi]
        update_w = (1.0 - self.momentum) * self.lr * dw - self.momentum * weights[wj]
        update_b = (1.0 - self.momentum_b) * self.lr_b * db - self.momentum_b * bias[wj]
        embeds[wi] += update_x
        weights[wj] += update_w
        bias[wj] += update_b


class AdaGrad(Trainer):
    def __init__(self, lr=.1, lr_b=None, epsilon=1e-6, gx_shape=None, gw_shape=None, gb_shape=None):
        super(AdaGrad, self).__init__(lr, lr_b)
        self.epsilon = numpy.array([epsilon], dtype=numpy.float32)
        self.acc_gx = zeros(gx_shape, dtype=numpy.float32)
        self.acc_gw = zeros(gw_shape, dtype=numpy.float32)
        self.acc_gb = zeros(gb_shape, dtype=numpy.float32)
        self.accumulator = None

    def compile(self, x, w, b, y, obj):
        acc_x = T.fmatrix('acc_x')
        acc_w = T.fmatrix('acc_w')
        acc_b = T.fvector('acc_b')

        obj_mean = T.mean(obj)
        obj_sum = T.sum(obj)
        gx, gw, gb = T.grad(obj_sum, [x, w, b])

        acc_nx = acc_x + gx ** 2
        acc_nw = acc_w + gw ** 2
        acc_nb = acc_b + gb ** 2

        gx_new = gx / T.sqrt(acc_x + self.epsilon)
        gw_new = gw / T.sqrt(acc_w + self.epsilon)
        gb_new = gb / T.sqrt(acc_b + self.epsilon)

        self.gradient = theano.function(inputs=[acc_x, acc_w, acc_b, x, w, b, y], outputs=[gx_new, gw_new, gb_new])
        self.accumulator = theano.function(inputs=[acc_x, acc_w, acc_b, x, w, b, y], outputs=[acc_nx, acc_nw, acc_nb])
        self.objective = theano.function(inputs=[x, w, b, y], outputs=obj_mean)

    def update(self, embeds, weights, bias, labels, wi, wj):
        self.acc_gx, self.acc_gw, self.acc_gb = self.accumulator(self.acc_gx, self.acc_gw, self.acc_gb,
                                                                 embeds[wi], weights[wj], bias[wj], labels)
        dx, dw, db = self.gradient(self.acc_gx, self.acc_gw, self.acc_gb,
                                   embeds[wi], weights[wj], bias[wj], labels)

        embeds[wi] += self.lr * dx
        weights[wj] += self.lr * dw
        bias[wj] += self.lr_b * db
