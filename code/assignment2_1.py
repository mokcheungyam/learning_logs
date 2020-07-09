# 1.1-sigmoid function
import math

def basic_sigmoid(x):
    return 1 / (1 + math.exp(-x))

# 使用math无法计算矩阵，因此使用numpy
import numpy as np

x = np.array([1, 2, 3])
np.exp(x)

x = np.array([1, 2, 3])

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 1.2-sigmoid gradient
def sigmoid_derivative(x):
    s = sigmoid(x)
    ds = s * (1 - s)

    return ds

# 1.3-reshaping arrays
# 三维变一维
def image2vecotr(image):
    v = image.reshape(image.shape[0] * image.shape[1] * image.shape[2], 1)

# 1.4-normalizing rows
# 正则化
def normalizerRows(x):
    x_norm = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / x_norm
    return x

# 1.5-broadcasting and the softmax function
def softmax(x):
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=1, keepdims=True)
    s = x_exp / x_sum
    return s

# vectorization
# 向量化提高运算效率

# 2.1-implement the l1 & l2 loss functions
def l1(yhat, y):
    loss = np.sum(np.abs(y - yhat))
    return loss

def l2(yhat, y):
    loss = np.dot((y - yhat), (y - yhat).T)
    return loss

# logistic regression with a neural network mindset

# packages we need
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset

# overview of the problem set
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

index = 5
plt.imshow(train_set_x_orig[index])

m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]

train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

train_set_x = train_set_x_flatten / 255.
test_set_x = test_set_x_flatten / 255.

# general architecture of the learning

# 4.1-helper functions
def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s

# 4.2-initializing parameters
def initializer_with_zeros(dim):
    w = np.zeros((dim, 1))
    b = 0

    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))

    return w, b

dim = 2
w, b = initializer_with_zeros(dim)

# 4.3-forward and backward propagation
def propagate(w, b, X, Y):
    m = X.shape[1]
    A = sigmoid(np.dot(w.T, X) + b)
    cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))

    dw = 1 / m * np.dot(X, (A - Y).T)
    db = 1 / m * np.sum(A - Y)

    assert(dw.shape == w.shape)
    assert(db.dtype == float)

    cost = np.squeeze(cost)  # np.squeeze()从数组的形状中删除单维度条目，将shape中为1的维度去掉
    assert(cost.shape == ())

    grads = {'dw': dw, 'db': db}

    return grads, cost

 def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    costs = []

    for i in range(num_iterations):

        grads, cost = propagate(w, b, X, Y)

        dw = grads['dw']
        db = grads['db']

        w = w - learning_rate * dw
        b = b - learning_rate * db

        if i % 100 == 0:
            costs.append(cost)

        if print_cost and i % 100 == 0:
            print('Cost after iteration %i: %f' % (i, cost))

    params = {'w': w, 'b': b}
    grads = {'dw': dw, 'db': db}

    return params, grads, costs

params, grads, costs = optimize(w, b, X, Y, num_iterations=100, learning=0.009, print_cost=0)


def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    A = sigmoid(np.dot(w.T, X) + b)

    for i in range(A.shape[1]):
        if A[0, i] <= 0.5:
            Y_prediction[0, i] = 0
        else:
            Y_prediction[0, i] = 1

    assert(Y_prediction.shape == (1, m))

    return Y_prediction


def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=0):
    w, b = initializer_with_zeros(X_train.shape[0])

    parameters, prads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)

    w = parameters['w']
    b = parameters['b']

    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    print('train accuracy: {} %'.format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print('test accuracy: {}%'.format(100 - np.mean(np.abs(Y_prediction_test - Y_train)) * 100))

    d = {'costs': costs, 'Y_prediction_test': Y_prediction_test,
         'Y_prediction_train': Y_prediction_train,
         'w': w,
         'b': b,
         'learning_rate': learning_rate,
         'num_iterations': num_iterations}
    return d
