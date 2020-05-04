import numpy as np

class ActivationLayer(object):
    def __init__(self, activation_func, d_activation_func):
        self.activation_func = activation_func
        self.d_activation_func = d_activation_func

    def forward_prop(self, X):
        self.X = X
        return self.activation_func(X)

    def backward_prop(self, d_global):
        return self.d_activation_func(self.X, d_global)



