import math


def threshold(k, *args):
    ''' Threshold'''
    theta = args[0]
    return (0 if k < theta else 1)


def logistic(k, *args):
    ''' Tanh'''
    theta = args[0]
    return 0.5 + 0.5 * math.tanh((k - theta)/2)


def relu(k, *args):
    ''' Relu '''
    theta = args[0]
    return max(0, k - theta)


ALLOWED_TYPES = {
    "threshold": threshold,
    "tanh": logistic,
    "relu": relu
}