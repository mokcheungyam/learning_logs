import numpy as np

def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    cache = Z

    return A, cache

def relu(Z):
    A = np.maximum(0, Z)

    assert(A.shape == Z.shape)

    cache = Z
    return A, cache

def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0

    assert (dZ.shape == Z.shape)

    return dZ

def sigmoid_backward(dA, cache):
    Z = cache

    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1-s)

    assert (dZ.shape == Z.shape)

    return dZ


import numpy as np

def linear_forward_test_case():
    np.random.seed(1)

    A = np.random.randn(3, 2)
    W = np.random.randn(1, 3)
    b = np.random.randn(1, 1)

    return A, W, b


def linear_acitvation_forward_test_case():
    np.random.seed(2)
    A_prev = np.random.randn(3, 2)
    W = np.random.randn(1, 3)
    b = np.random.randn(1, 1)
    return A_prev, W, b

def L_model_forward_test_case():
    np.random.seed(1)
    X = np.random.randn(4, 2)
    W1 = np.random.randn(3, 4)
    b1 = np.random.randn(3, 1)
    W2 = np.random.randn(1, 3)
    b2 = np.random.randn(1, 1)
    parameters = {'W1': W1,
                  'b1': b1, 
                  'W2': W2,
                  'b2': b2}

    return X, parameters

def compute_cost_test_case():
    Y = np.asarray([[1, 1, 1]])
    aL = np.array([[.8, .9, .4]])
    return Y, aL

def linear_backward_test_case():
    np.random.seed(2)
    dA = np.random.randn(1, 2)
    A = np.random.randn(3, 2)
    W = np.random.randn(1, 3)
    b = np.random.randn(1, 1)
    Z = np.random.randn(1, 2)
    linear_cache = (A, W, b)
    activation_cache = Z
    linear_activation_cache = (linear_cache, activation_cache)

    return dA, linear_activation_cache

def L_model_backward_test_case():
    np.random.seed(3)
    AL = np.random.randn(1, 2)
    Y = np.array([[1, 0]])

    A1 = np.random.randn(4, 2)
    W1 = np.random.randn(3, 4)
    b1 = np.random.randn(3, 1)
    Z1 = np.random.randn(3, 2)
    linear_cache_activation_1 = ((A1, W1, b1), Z1)

    A2 = np.random.randn(3, 2)
    W2 = np.random.randn(1, 3)
    b2 = np.random.randn(1, 1)
    Z2 = np.random.randn(1, 2)
    linear_cache_activation_2 = ((A2, W2, b2), Z2)

    caches = (linear_cache_activation_1, linear_cache_activation_2)

    return AL, Y, caches

def update_parameters_test_case():
    np.random.seed(2)
    W1 = np.random.randn(3, 4)
    b1 = np.random.randn(3, 1)
    W2 = np.random.randn(1, 3)
    b2 = np.random.randn(1, 1)
    parameters = {'W1': W1,
                  'b1': b1,
                  'W2': W2,
                  'b2': b2}
    np.random.seed(3)
    dW1 = np.random.randn(3, 4)
    db1 = np.random.randn(3, 1)
    dW2 = np.random.randn(1, 3)
    db2 = np.random.randn(1, 1)
    grads = {'dW1': dW1,
             'db1': db1,
             'dW2': dW2,
             'db2': db2}
    return parameters, grads

# ....








# Building your Deep Neural Network
# 1.Packages
import numpy as np
import h5py
import matplotlib.pyplot as plt
from testCase_v2 import *
from dnn_utils_v2 import sigmoid, sigmoid_backward, relu, relu_backward

# set default size of plots
plt.rcParams['figure.figsize'] = (5.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)

# 2.outline of the assignment

# 3.initialization
# 3.1 2-layer Neural Network
def initialize_parameters(n_x, n_h, n_y):
    np.random.seed(1)

    W1 = None
    b1 = None
    W2 = None
    b2 = None

    assert(W1.shape == (n_h, n_x))
    assert(b1.shape == (n_h, 1))
    assert(W2.shape == (n_y, n_h))
    assert(b2.shape == (n_y, 1))

    parameters = {'W1': W1,
                  'b1': b1,
                  'W2': W2,
                  'b2': b2}

    return parameters

parameters = initializer_parameters(2, 2, 1)

# 3.2 
def initialize_parameters_deep(layer_dims):
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
        parameters['W' + str(l)] = None
        parameters['b' + str(l)] = None

        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

    return parameters

parameters = initialize_parameters_deep([5, 4, 3])

# 4.forward propagtion module
# 4.1.linear forward
def linear_forward(A, W, b):
    Z = None

    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)

    return Z, cache

A, W, b = linear_forward_test_case()

z, linear_cache = linear_forward(A, W, b)

# 4.2.linear-activation forward
def linear_activation_forward(A_prev, W, b, activation):
    if activation == 'sigmoid':
        Z, linear_cache = None
        A, activation_cache = None

    elif activation == 'relu':
        Z, linear_cache = None
        A, activation_cache = None

    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache

A_prev, W, b = linear_acitvation_forward_test_case()

A, linear_forward_test_cache = linear_activation_forward(A_prev, W, b, activation='sigmoid')

A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation='relu')


def L_model_forward(X, prameters):
    caches = []
    A = X
    L = len(parameters) // 2

    for l in range(1, L):
        A_prev = A_prev
        A, cache = None

    AL, cache = None

    assert(AL.shape == (1, X.shape[1]))
    return AL, caches

X, prameters = L_model_forward_test_case()

AL, caches = L_model_forward(X, parameters)


# 5.cost function
def compute_cost(AL, Y):
    m = Y.shape[1]

    cost = None

    cost = np.squeeze(cost)
    assert(cost.shape == ())

    return cost

Y, AL = compute_cost_test()

# 6.backward propagation module
# 6.1.linear backward
def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = None
    db = None
    dA_prev = None

    assert(dA_prev.shape == A_prev.shape)
    assert(dW.shape == W.shape)
    assert(db.shape == b.shape)

    return dA_prev, dW, db

dZ, linear_cache = linear_backward_test_case()

dA_prev, dW, db = linear_forward(dZ, linear_cache)

# 6.2.linear-activation backward
def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache
    if activation == 'relu':
        dZ = None
        dA_prev, dW, db = None

    elif activation == 'sigmoid':
        dZ = None
        dA_prev, dW, db = None

    return dA_prev, dW, db

AL, linear_activation_cache = linear_activation_backward_test_case()

dA_prev, dW, db = linear_activation_backward(AL, linear_activation_cache, activation='sigmoid')
dA_prev, dW, db = linear_activation_backward(AL, linear_activation_cache, activation='relu')


# 6.3 model backward
def L_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)    

    dAL = None

    current_cache = None
    grads['dA' + str(L)], grads['dW' + str(L)], grads['db' + str(L)] = None
    for l in reversed(range(L - 1)):
        current_cache = None
        dA_prev_temp, dW_temp, db_temp = None
        grads['dA' + str(l + 1)] = None
        grads['dW' + str(l + 1)] = None
        grads['db' + str(l + 1)] = None

    return grads

AL, Y_assess, caches = L_model_backward_test_case()
grads = L_model_backward(AL, Y_assess, caches)

# 6.4 update parameters
def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    for l in range(L):
        parameters['W' + str(l+1)] = None
        parameters['b' + str(l+1)] = None
    return parameters

parameters, grads = update_parameters_test_case()
parameters = update_parameters(parameters, grads, 0.1)

# 7.conclusion

