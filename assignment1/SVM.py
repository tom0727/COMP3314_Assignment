import numpy as np
from cvxopt import matrix,solvers
import matplotlib.pyplot as plt
import pandas as pd
plt.style.use('ggplot')

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

        P = matrix(self.K * np.outer(y,y))
        q = matrix(np.ones(n) * -1.0)  # q = [-1,-1,-1,...]

        negI, I = np.identity(n) * -1.0, np.identity(n)

        G = matrix(np.vstack((negI,I)))  # G = [-I,I]^T
        h = matrix(np.hstack((np.zeros(n), np.ones(n) * self.C)))  # h = [0,0,...0, C,C,..C]

        # With G and h, we make sure 0 <= a_i <= C

        A = matrix(y, (1,n), 'd')  # A = [y1,y2,y3,...], since Aa = 0
        b = matrix(0.0)

        sol = solvers.qp(P,q,G,h,A,b)
        eps = 1e-6

        a = np.array(sol["x"]).reshape(n)  # a is a vector containing the lagrange multipliers
        mask = (a > eps)  # a mask containing [T,T,F,...]
        indices = np.arange(0,n)[mask]  # get the indices of non-zero multipliers

        self.X = X[mask]
        self.y = y[mask]
        self.a = a[mask]
        self.num = len(self.a)  # the number of non-zero multipliers

        self.w = np.zeros(m)
        for i in range(self.num):
            self.w += (self.a[i] * self.y[i] * self.X[i])

        self.b = 0.0
        for i in range(self.num):
            self.b += self.y[i]
            for j in range(self.num):
                self.b -= self.a[j] * self.y[j] * self.K[indices[i],indices[j]]
        # b = sum_i (y_i - sum_j (a_j * y_j (x_i * x_j)))
        self.b /= self.num

    def predict(self, X):
        predicted = np.dot(X, self.w) + self.b
        return predicted

def process_data(x, label):  # classify label into 1, others into -1
    tmp = np.ravel(x)
    tmp = np.array([-1 if i != label else 1 for i in tmp])
    return tmp.reshape(x.shape)

def cancer_data(C):
    X_test = pd.read_csv("dataset_files/cancer_X_test.csv").values
    X_train = pd.read_csv("dataset_files/cancer_X_train.csv").values
    y_test = pd.read_csv("dataset_files/cancer_y_test.csv").values.ravel()
    y_train = pd.read_csv("dataset_files/cancer_y_train.csv").values.ravel()
    y_test = process_data(y_test, 0)
    y_train = process_data(y_train, 0)

    svm = SVM(C)
    svm.fit(X_train, y_train)
    y = svm.predict(X_test)  # prediction result
    y = np.sign(y)  # binary classification
    res = np.sum(y == y_test)

    print(f"Predicted {res}/{len(y_test)} correctly, accuracy = {res/len(y_test) * 100}%")
    return res / len(y_test) * 100

def iris_data(C):
    X_test = pd.read_csv("dataset_files/iris_X_test.csv").values
    X_train = pd.read_csv("dataset_files/iris_X_train.csv").values
    y_test = pd.read_csv("dataset_files/iris_y_test.csv").values.ravel()
    y_train = pd.read_csv("dataset_files/iris_y_train.csv").values.ravel()

    y_train_0 = process_data(y_train, 0)
    svm_0 = SVM(C)
    svm_0.fit(X_train, y_train_0)
    y_0 = svm_0.predict(X_test)

    y_train_1 = process_data(y_train, 1)
    svm_1 = SVM(C)
    svm_1.fit(X_train, y_train_1)
    y_1 = svm_1.predict(X_test)

    y_train_2 = process_data(y_train, 2)
    svm_2 = SVM(C)
    svm_2.fit(X_train, y_train_2)
    y_2 = svm_2.predict(X_test)

    tmp = np.array([y_0,y_1,y_2])
    y = np.argmax(tmp, axis = 0)
    res = np.sum(y == y_test)
    # Find the class with the highest score, assign to that class

    print(f"Predicted {res}/{len(y_test)} correctly, accuracy = {res/len(y_test) * 100}%")
    return res/len(y_test) * 100

def plot_result():
    c_values = []
    for i in [0.001,0.01,0.1,1.0,10]:
        for j in range(1,10):
            c_values.append(i*j)

    carcer_accuracy = [cancer_data(c) for c in c_values]
    iris_accuracy = [iris_data(c) for c in c_values]

    fig = plt.figure()
    ax1 = plt.subplot2grid((1,1),(0,0))
    ax1.plot(c_values,carcer_accuracy,label='breast cancer')
    ax1.plot(c_values,iris_accuracy, label='iris')
    ax1.set_xscale('log')
    plt.legend()
    plt.xlabel('Slack variable values')
    plt.ylabel('Accuracy (in %)')
    plt.title('Accuracy of two classification under different slack variables')
    plt.subplots_adjust(left=0.09,bottom=0.16,right=0.94,top=0.9,wspace=0.2,hspace=0)
    plt.show()

plot_result()