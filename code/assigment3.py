import numpy as np
import matplotlib.pyplot as plt
from testCases import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets


X, Y = load_planar_dataset()

shape_X = None
shape_Y = None
m = shape_X[1]

# simple logistic regression
# train the logistic regression classifier
clf = sklearn.linear_model.LogisticRegressionCV()
clf.fit(X.T, Y.T)


plot_decision_boundary(lambda x: clf.predict(x), X, Y)
plt.title('Logistic Regression')

LR_predictions = clf.predict(X.T)

# neural network model
# 1.defining the neural network structure
def layer_sizes(X, Y):
    n_x = None
    n_h = None
    n_y = None
    return (n_x, n_h, n_y)

X_assess, Y_assess = layer_sizes_test_case()
(n_x, n_h, n_y) = layer_sizes(X_assess, Y_assess)

# 2.initialize the model's parameters
def initializer_parameters(n_x, n_h, n_y):
    np.random.seed(2)

    W1 = None
    b1 = None
    W2 = None
    b2 = None

    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))

    parameters = {'W1': W1,
                  'b1': b1,
                  'W2': W2,
                  'b2': b2}

    return parameters

n_x, n_h, n_y = initializer_parameters_test_case()

parameters = initializer_parameters(n_x, n_h, n_y)

# 3.the loop
def forward_propagation(X, parameters):
    W1 = None
    b2 = None
    W2 = None
    b2 = None

    z1 = None
    A1 = None
    z2 = None
    A2 = None

    assert(A2.shape == (1, X.shape[1]))

    cache = {'Z1': Z1,
             'A1': A1,
             'Z2': Z2,
             'A2': A2}
    return A2, cache

X_assess, parameters = forward_propagation_test_case()

A2, cache = forward_propagation(X_assess, parameters)

def compute_cost(A2, Y, parameters):
    m = Y.shape[1]
    logprobs = None
    cost = None

    cost = np.squeeze(cost)
    assert(isinstance(cost, float))
    return cost

A2, Y_assess, parameters = compute_cost_test_case()

def backward_propagation(parameters, cache, X, Y):
    m = X.shape[1]
    W1 = None
    W2 = None

    A1 = None
    A2 = None

    dz2 = None
    dw2 = None

    db2 = None
    dz1 = None
    db1 = None

    grads = {'dw1': dw1,
             'db1': db1,
             'dw2': dw2, 
             'db2': db2}
    return grads

parameters, cache, X_assess, Y_assess = backward_propagation_test_case()
grads = backward_propagation(parameters, cache, X_assess, Y_assess)

# 白抄一遍题目
