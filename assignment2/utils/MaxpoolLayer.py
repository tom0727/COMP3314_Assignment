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

        # assuming stride = filtersize
        Y_repeat = self.Y.repeat(self.filtersize, axis = 2).repeat(self.filtersize, axis = 3)
        mask = np.equal(Y_repeat, self.X)
        # (n_sample, depth, in_h, in_w)
        return mask * (d_global.repeat(self.filtersize, axis = 2).repeat(self.filtersize, axis = 3))
