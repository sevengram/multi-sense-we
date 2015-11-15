import theano
from theano import tensor as T


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

    def compile(self, x, w, b, y, obj_func):
        gx, gw, gb = T.grad(obj_func, [x, w, b])
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
    def __init__(self, lr=.1, lr_b=None, epsilon=1e-6):
        super(AdaGrad, self).__init__(lr, lr_b)
        self.epsilon = epsilon

    def compile(self, x, w, b, y, obj_func):
        #TODO
        pass

    def update(self, wi, wj, embeds, labels, weights, bias, gradients):
        #TODO
        pass