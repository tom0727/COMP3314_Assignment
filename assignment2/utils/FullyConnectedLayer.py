import numpy as np
from utils.helper import *

class FullyConnectedLayer:

    def __init__(self, weight_shape):
        """
        :param weight_shape: two-dimensional. F6: (120,84). F7: (84,10)
        """
        self.W = np.random.normal(0, 0.1, weight_shape)
        self.b = np.ones((weight_shape[-1], )) * 0.01  # (84,) or (10,)
        self.v_W = np.zeros(self.W.shape)
        self.v_b = np.zeros(self.b.shape)

    def forward_prop(self, X):
        self.X = X
        return np.dot(X, self.W) + self.b

    def backward_prop(self, d_global, learning_rate, momentum = 0.9):
        """
        :param d_global: global derivative passed to this gate, shape = (n_sample, 10) or (n_sample, 84)
        :param learning_rate: global learning rate
        :return: dX: The derivative wrt the input matrix X
        """
        db = np.sum(d_global, axis = 0)  # db.shape = (10,)
        dX = np.dot(d_global, self.W.T)  # (n_sample, 10) * (10, 84) = (n_sample, 84)
        dW = np.dot(self.X.T, d_global)  # (84, n_sample) * (n_sample, 10) = (84, 10)

        self.W, self.b, self.v_W, self.v_b = update(self.W, self.b, dW, db, self.v_W, self.v_b, learning_rate, momentum)
        return dX
