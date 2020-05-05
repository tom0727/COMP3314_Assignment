import numpy as np

# To calculate loss, we must be in the 'train' mode
class LossLayer:
    def __init__(self, label):
        self.label = label  # shape = (n_sample, 1)

    def forward_prop(self, X):
        """
        :param X: After F7, shape = (n_sample, 10)
        :return: a scalar, loss
        """
        n_sample = X.shape[0]

        X_exp = np.exp(X - np.max(X, axis = 1).reshape(n_sample,1))
        self.X_softmax = X_exp / (np.sum(X_exp, axis = 1).reshape((n_sample,1)))

        prob = self.X_softmax[np.arange(n_sample), self.label]  # prob.shape = (n_sample, )
        log_prob = np.log(prob + 1e-10)  # avoid 0.

        loss = -np.sum(log_prob)
        return loss

    def backward_prop(self):
        n_sample, n_label = self.X_softmax.shape
        label_mat = np.zeros((n_sample, n_label))
        label_mat[np.arange(n_sample), self.label] -= 1
        return label_mat + self.X_softmax

