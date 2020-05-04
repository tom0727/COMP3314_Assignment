import numpy as np

class MaxpoolLayer:
    def __init__(self, filtersize = 2):
        self.filtersize = filtersize
        self.stride = filtersize  # assume stride = filtersize

    def forward_prop(self, X):
        """
        :param X: input matrix X, shape = (n_sample, depth, in_h, in_w)
        :return: Output matrix Y, shape = (n_sample, depth, out_h, out_w);  out_h = in_h / 2
        """
        self.X = X
        n_sample, depth, in_h, in_w = X.shape

        out_h = int((in_h - self.filtersize) // self.stride + 1)
        out_w = int((in_w - self.filtersize) // self.stride + 1)

        Y = np.zeros((n_sample, depth, out_h, out_w))

        for i in range(out_h):
            for j in range(out_w):
                X_slice = X[:,:, (i*self.stride):(i*self.stride+self.filtersize), (j*self.stride):(j*self.stride+self.filtersize)]
                Y[:,:,i,j] = np.max(X_slice, axis = (2,3))

        assert(Y.shape == (n_sample, depth, out_h, out_w))

        self.Y = Y  # store the maxpool result
        return Y

    def backward_prop(self, d_global):
        n_sample, depth, out_h, out_w = d_global.shape
        # in_h = (out_h - 1) * self.stride + self.filtersize
        # in_w = (out_w - 1) * self.stride + self.filtersize

        # assuming stride = filtersize
        Y_repeat = self.Y.repeat(self.filtersize, axis = 2).repeat(self.filtersize, axis = 3)
        # print(f"Y_repeat =\n {Y_repeat}")
        # print(f"self.X =\n {self.X}")
        # print(f"self.X.shape = {self.X.shape}")
        # print(f"self.Y.shape = {self.Y.shape}")
        # print(f"Y_repeat.shape = {Y_repeat.shape}")
        mask = np.equal(Y_repeat, self.X)
        # print(f"mask = \n {mask}")

        # (n_sample, depth, in_h, in_w)
        return mask * (d_global.repeat(self.filtersize, axis = 2).repeat(self.filtersize, axis = 3))

    def max_pool_backward_naive(self, dout):
        """
        A naive implementation of the backward pass for a max pooling layer.
        Inputs:
        - dout: Upstream derivatives
        - cache: A tuple of (x, pool_param) as in the forward pass.
        Returns:
        - dx: Gradient with respect to x
        """
        dx = None
        x = self.X
        HH, WW, stride = 2,2,2
        N, C, H, W = x.shape
        # HH, WW, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
        H_prime = (H - HH) // stride + 1
        W_prime = (W - WW) // stride + 1
        dx = np.zeros_like(x)
        for n in range(N):
            for c in range(C):
                for i in range(H_prime):
                    h_start = i * stride
                    for j in range(W_prime):
                        w_start = j * stride
                        local_x = x[n, c, h_start:h_start + HH, w_start:w_start + WW]
                        h, w = np.unravel_index(local_x.argmax(), (HH, WW))
                        dx[n, c, h_start + h, w_start + w] += dout[n, c, i, j]
        return dx

# layer = MaxpoolLayer()
# X = np.random.randint(-5,25,(2,2,4,4))
# print(f"X = {X}")
# print()
# print()
# print()
# print()
# y = layer.forward_prop(X)
# print(f"y = {y}")
# print()
# print("Backward result = ")
# print(layer.backward_prop(y))
# print("correct version: ")
# print(layer.max_pool_backward_naive(y))
