import numpy as np
from utils.helper import *

class ConvolveLayer:

    def __init__(self, kernel_shape, stride = 1, pad = 0):
        """
        :param kernel_shape: The shape of matrix W, shape = (n_filter, depth, filtersize, filtersize)
        :param stride: Assume stride = 1 in back_prop
        :param padding:
        """

        self.kernel_shape = kernel_shape
        self.n_filter, self.depth, self.filtersize, _ = kernel_shape

        self.stride = stride
        self.pad = pad
        # self.W = np.random.normal(0, 0.1, kernel_shape)
        self.b = np.ones((1,self.n_filter,1,1)) * 0.01  # (1,n_filter,1,1)

        fan_in = np.prod(kernel_shape) / kernel_shape[0]
        self.W = np.random.uniform(-2.4/fan_in, 2.4/fan_in, kernel_shape)

        self.v_W, self.v_b = np.zeros(self.W.shape), np.zeros(self.b.shape)

    def reinitialize(self):
        fan_in = np.prod(self.kernel_shape) / self.kernel_shape[0]
        self.W = np.random.uniform(-2.4/fan_in, 2.4/fan_in, self.kernel_shape)

    def forward_prop(self, X):
        """
        :param X: Input matrix X, shape = (n_sample, depth, in_h, in_w)
        :return: Output matrix Y, shape = (n_sample, n_filter, out_h, out_w)
        """
        # Usually, in_h = in_w, out_h = out_w

        n_sample, depth, in_h, in_w = X.shape


        out_h = int((in_h + 2*self.pad - self.filtersize) // self.stride + 1)
        out_w = int((in_w + 2*self.pad - self.filtersize) // self.stride + 1)

        Y = np.zeros((n_sample, self.n_filter, out_h, out_w))

        X_pad = np.pad(X, ((0,), (0,), (self.pad,), (self.pad,)), 'constant', constant_values = (0,0))
        self.X = X_pad

        for i in range(out_h):
            for j in range(out_w):
                X_slice = X_pad[:,:, (i*self.stride):(i*self.stride+self.filtersize), (j*self.stride):(j*self.stride+self.filtersize)]
                Y[:,:,i,j] = np.tensordot(X_slice, self.W, [[1,2,3],[1,2,3]])  # Y[:,:,i,j].shape = (n_sample,n_filter)

        Y += self.b  # (n_sample, n_filter, out_h, out_w) + (1,n_filter,1,1)

        assert(Y.shape == (n_sample, self.n_filter, out_h, out_w))
        return Y


    def backward_prop(self, d_global, learning_rate, momentum = 0.9):
        """
        :param d_global: global derivative passed to this gate, shape = (n_sample, 16, 10, 10)
        :return: dX: The derivative wrt the input matrix X
        """
        n_sample, _, out_h, out_w = d_global.shape  # out_h, out_w are the dimension of the global gradient
        d_global_pad = np.pad(d_global, ((0,), (0,), (self.filtersize - 1,), (self.filtersize - 1,)), 'constant', constant_values = (0,0))
        # d_global_pad is used to calculate convolution with flipped Weight

        dX = np.zeros((n_sample, self.depth, out_h + self.filtersize - 1, out_w + self.filtersize - 1))
        dW = np.zeros(self.W.shape)
        db = np.zeros(self.b.shape)

        in_h = dX.shape[2]
        in_w = dX.shape[3]
        out_h_pad = d_global_pad.shape[2]
        out_w_pad = d_global_pad.shape[3]

        W_f = self.W[:,:,::-1,::-1]  # The flipped version of W

        for i in range(in_h):
            for j in range(in_w):
                d_global_slice = d_global_pad[:,:,i:i+self.filtersize,j:j+self.filtersize]
                dX[:,:,i,j] = np.tensordot(d_global_slice, W_f, [[1,2,3],[0,2,3]])

        for i in range(self.filtersize):
            for j in range(self.filtersize):
                X_slice = self.X[:,:,i:i+out_h,j:j+out_w]  # shape = (n,16,10,10)
                dW[:,:,i,j] = np.tensordot(d_global, X_slice, [[0,2,3],[0,2,3]])  # shape = (16,6)

        db[0,:,0,0] = np.sum(d_global, axis = (0,2,3))
        self.W, self.b, self.v_W, self.v_b = update(self.W, self.b, dW, db, self.v_W, self.v_b, learning_rate, momentum)

        return dX