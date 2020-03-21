import numpy as np
from cvxopt import matrix,solvers

class SVM:
    def __init__(self, C = 1.0):
        self.C = C  # slack penalty

    def fit(self, X, y):

        # N is n*m, y is n (one dimensional)

        n,m = X.shape  # n is the number of samples, m is the number of features

        self.K = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                self.K[i,j] = X[i].dot(X[j])  # K is the kernel matrix, the value is used later on


        P = matrix(np.dot(self.K, np.outer(y,y)))
        q = matrix(np.ones(n) * -1.0)  # q = [-1,-1,-1,...]

        negI, I = np.identity(n) * -1.0, np.identity(n)
        G = matrix(np.vstack(negI,I))  # G = [-I,I]^T

        h = matrix(np.hstack(np.zeros(n), np.ones(n)*C))  # h = [0,0,...0, C,C,..C]

        # With G and h, we make sure 0 <= a_i <= C

        A = matrix(y, (1,n))  # A = [y1,y2,y3,...], since Aa = 0
        b = matrix(0.0)

        sol = solvers.qp(P,q,G,h,A,b)
        eps = 1e-6

        a = sol["x"].reshape(n)  # a is a vector containing the lagrange multipliers
        mask = (a > eps)  # a mask containing [T,T,F,...]
        indices = np.arange(0,n)[mask]  # get the indices of non-zero multipliers

        self.X = X[mask]
        self.y = y[mask]
        self.a = a[mask]
        self.num = len(self.a)  # the number of non-zero multipliers

        self.w = np.zeros(n)
        for i in range(self.num):
            self.w += (self.a[i] * self.y[i] * self.X[i])

        self.b = 0.0
        for i in range(self.num):
            self.b += self.y[i]
            for j in range(self.num):
                self.b -= self.a[j] * self.y[j] * self.K[indices[i],indices[j]]
                # self.b -= (np.sum(self.a * self.y * (K[indices[i], mask])))
        # since b = sum_i (y_i - sum_j (a_j * y_j (x_i * x_j)))
        self.b /= self.num


    def predict(self, X):  # X is still a matrix consisting of many samples
        n = len(X)  # number of samples
        y = np.zeros(n)
        for i in range(n):
            res = self.w.dot(X[i]) + self.b  # wx+b
            y[i] = np.sign(res)  # y[i] = 1 or 0 or -1
        return y






















