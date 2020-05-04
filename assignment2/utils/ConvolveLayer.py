import numpy as np
from utils.helper import *

class ConvolveLayer:

    def __init__(self, kernel_shape, in_shape, stride = 1, pad = 0):
        """
        :param kernel_shape: The shape of matrix W, shape = (n_filter, depth, filtersize, filtersize)
        :param stride: Assume stride = 1 in back_prop
        :param padding:
        """

        self.n_filter, self.depth, self.filtersize, _ = kernel_shape

        self.stride = stride
        self.pad = pad
        # self.W = np.random.normal(0, 0.1, kernel_shape)
        self.b = np.ones((1,self.n_filter,1,1)) * 0.01  # (1,n_filter,1,1)

        fan_in = np.prod(kernel_shape) / kernel_shape[0]
        self.W = np.random.uniform(-2.4/fan_in, 2.4/fan_in, kernel_shape)
        # fan_out = self.n_filter * np.prod(kernel_shape[2:])

        self.v_W, self.v_b = np.zeros(self.W.shape), np.zeros(self.b.shape)

    def forward_prop(self, X):
        """
        :param X: Input matrix X, shape = (n_sample, depth, in_h, in_w)
        :return: Output matrix Y, shape = (n_sample, n_filter, out_h, out_w)
        """
        # Usually, in_h = in_w, out_h = out_w

        n_sample, depth, in_h, in_w = X.shape

        # print(f"in_h = {in_h}")
        # print(f"filtersize = {self.filtersize}")

        out_h = int((in_h + 2*self.pad - self.filtersize) // self.stride + 1)
        out_w = int((in_w + 2*self.pad - self.filtersize) // self.stride + 1)

        # print(f"out_h = {out_h}")
        # print(f"out_w = {out_w}")

        Y = np.zeros((n_sample, self.n_filter, out_h, out_w))

        X_pad = np.pad(X, ((0,), (0,), (self.pad,), (self.pad,)), 'constant', constant_values = (0,0))
        self.X = X_pad

        for i in range(out_h):
            for j in range(out_w):
                X_slice = X_pad[:,:, (i*self.stride):(i*self.stride+self.filtersize), (j*self.stride):(j*self.stride+self.filtersize)]
                # if (i % 3 == 0 and j % 3 == 0):
                #     print(f"X_slice = {X_slice}")

                # print(f"X_slice.shape = {X_slice.shape}")
                # print(f"b.shape = {self.b.shape}")
                # print(f"tensordot(X_slice,W).shape = {np.tensordot(X_slice, self.W, [[1,2,3],[1,2,3]]).shape}")
                Y[:,:,i,j] = np.tensordot(X_slice, self.W, [[1,2,3],[1,2,3]])  # Y[:,:,i,j].shape = (n_sample,n_filter)

        Y += self.b  # (n_sample, n_filter, out_h, out_w) + (1,n_filter,1,1)

        # print(f"Y = \n{Y}")

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

        # for d1 in range(self.depth):
        #     dX[:,d1,:,:] = np.tensordot(d_global[:,], W_flipped)

        W_f = self.W[:,:,::-1,::-1]  # The flipped version of W

        for i in range(in_h):
            for j in range(in_w):
                d_global_slice = d_global_pad[:,:,i:i+self.filtersize,j:j+self.filtersize]
                dX[:,:,i,j] = np.tensordot(d_global_slice, W_f, [[1,2,3],[0,2,3]])
        # print(f"mydx = \n{dX}")
        # dX = self.conv_backward_naive(dglobal)[0]


        # print(f"self.X = \n{self.X}")
        for i in range(self.filtersize):
            for j in range(self.filtersize):
                X_slice = self.X[:,:,i:i+out_h,j:j+out_w]  # shape = (n,16,10,10)
                dW[:,:,i,j] = np.tensordot(d_global, X_slice, [[0,2,3],[0,2,3]])  # shape = (16,6)

        # print(f"d_global =\n {d_global}")
        # print(f"mydW =\n {dW}")
        # print(f"dX =\n {dX}")

        db[0,:,0,0] = np.sum(d_global, axis = (0,2,3))
        self.W, self.b, self.v_W, self.v_b = update(self.W, self.b, dW, db, self.v_W, self.v_b, learning_rate, momentum)

        # print(f"self.W = \n{self.W}")

        return dX


    def conv_backward_naive(self,dout):
      """
      A naive implementation of the backward pass for a convolutional layer.
      Inputs:
      - dout: Upstream derivatives.
      - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive
      Returns a tuple of:
      - dx: Gradient with respect to x
      - dw: Gradient with respect to w
      - db: Gradient with respect to b
      """
      dx, dw, db = None, None, None

      # (x, w, b, conv_param) = cache
      x = self.X
      w = self.W
      b = self.b
      N, C, H, W = x.shape
      F, _, HH, WW = w.shape
      dw, db = np.zeros_like(w), np.zeros_like(b)
      stride = 1
      pad = 0
      dx = np.zeros((N, C, H + 2 * pad, W + 2 * pad))
      H_prime = 1 + (H + 2 * pad - HH) // stride
      W_prime = 1 + (W + 2 * pad - WW) // stride

      for n in range(N):
        x_n = np.pad(x[n], ((0,), (pad,), (pad,)), 'constant')  # (C, H+pad, W+pad)
        for i in range(H_prime):
          h_start = i * stride
          for j in range(W_prime):
            w_start = j * stride
            for f in range(F):
              dx[n, :, h_start:h_start + HH, w_start:w_start + WW] += w[f] * dout[n, f, i, j]
              dw[f] += x_n[:, h_start:h_start + HH, w_start:w_start + WW] * dout[n, f, i, j]
      # db = dout.sum((0, 2, 3))

      # dx = dx[:, :, pad:-pad, pad:-pad]
      # print(f"dx_correct = {dx}")
      # print(f"dw_correct = {dw}")

      return dx, dw, db




# layer = ConvolveLayer((2,2,3,3))
# layer.W = np.linspace(-3,5,9).reshape((1,1,3,3))
# layer.W = np.linspace(-6,10,36).reshape((2,2,3,3))

# print(f"layer.W = \n{layer.W}")
# layer.b *= 0

# X = np.linspace(1,16,16).reshape((1,1,4,4))
# X = np.linspace(-15,30,64).reshape((2,2,4,4))
# y = layer.forward_prop(X)

# dglobal = np.linspace(1,4,4).reshape((1,1,2,2))
# dglobal = np.linspace(-9,8,16).reshape((2,2,2,2))
# layer.backward_prop(dglobal, 0.5)
# layer.conv_backward_naive(dglobal)

# testlayer = Conv3x3(1)
# testlayer.filters = np.linspace(-3,5,9).reshape((1,3,3))
# testlayer.forward(X[0][0])
# testlayer.backprop(dglobal[0].reshape((2,2,1)), 0.5)

# convBack(dglobal[0], X[0], layer.W)
