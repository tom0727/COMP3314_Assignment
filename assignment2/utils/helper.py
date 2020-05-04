import numpy as np

def ReLU(x):
    return np.where(x>0, x, 0)

def d_ReLU(x, d_global):
    mask = (x > 0)
    return mask * d_global

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def update(W, b, dW, db, v_W, v_b, learning_rate, momentum = 0.9):
    incre_v_W = momentum * v_W - learning_rate * dW
    incre_v_b = momentum * v_b - learning_rate * db
    return W + incre_v_W, b + incre_v_b, incre_v_W, incre_v_b

