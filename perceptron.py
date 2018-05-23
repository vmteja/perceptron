import math
import random
import re
from operator import mul
import sys


def threshold(k):
    ''' Threshold activation function'''
    return (0 if k < theta else 1)


def sigmoid(k):
    '''Sigmoid activation function '''
    return 1/(1+math.exp(-k))


def tanh(k):
    ''' Tanh activation function '''
    return math.tanh(k)


def relu(k):
    ''' ReLu activation function '''
    return max(0, k)

activation = {"threshold": threshold,
              "tanh": tanh,
              "sigmoid": sigmoid,
              "relu": relu}


def perceptron(x, w, activation, theta):
    '''Takes an input, weights
    applies an activation on the linear product of inputs and weights
    and returns 1 if the output of activation is greater than theta
    if not it returns 0 '''

    linear_sum = sum(i*j for i, j in zip(x, w))

    if activation(linear_sum) >= theta:
        return 1
    else:
        return 0


def perceptron_update_rule(y_true, y_predicted, x, w, theta):
    

    if y_predicted > y_true:

        temp_w = [j - i for i, j in zip(x, w)]
        temp_theta = theta + 1
        w = temp_w
        theta = temp_theta
        print x, ":", y_true, ":", "update"

    if y_predicted < y_true:

        temp_w = [i+j for i, j in zip(x, w)]
        temp_theta = theta - 1
        w = temp_w
        theta = temp_theta
        print x, ":", y_true, ":", "update"

    else:
        print x, ":", y_true, ":", "no update"

    return w, theta


def winnow_update_rule(y_true, y_predicted, x, w, alpha=1.1, theta=0.5):
    

    if y_predicted > y_true:

        temp_w = [(alpha ** -i) * j for i, j in zip(x, w)]
        w = temp_w
        print x, ":", y_true, ":", "update"

    elif y_predicted < y_true:

        temp_w = [(alpha ** i) * j for i, j in zip(x, w)]
        w = temp_w
        print x, ":", y_true, ":", "update"

    else:
        print x, ":", y_true, ":", "no update"

    return w, theta


def compute_ground_tf(X, coeffs, th):
    

    return [int(sum(map(mul, x, coeffs)) >= th) for x in X]


def funapply(num1, num2, fun):
    

    if fun == 'OR':
        return num1 or num2
    elif fun == 'AND':
        return num1 and num2


def compute_ground_nbf(X, sign_indexes, funs):
    
    y = []
    for x in X:
        nums = list(map(
                   lambda i: (1 if i > 0 else -1) * x[abs(i) - 1],
                    sign_indexes))
        num1 = nums[0]
        for i in range(len(funs)):
            num2 = nums[i + 1]
            num1 = funapply(num1, num2, funs[i])
        y.append(num1)
    return y


def unit_sphere_distribution_sample(size):
    
    sphere_distribution = [random.random() for i in range(size)]

    mag = math.sqrt(sum(i ** 2 for i in sphere_distribution))

    sphere = [i / mag for i in sphere_distribution]

    return sphere


def boolean_distribution_sample(size):
    
    return [random.randint(0, 1) for i in range(size)]


def train(X_train, y_train):
    ''' Training the perceptron'''

    w = [1] * len(X_train[0])
    error = 0
    theta = 0
    for i in range(num_train):

        y_predicted = perceptron(X_train[i], w, activation_func, theta)
        y_true = y_train[i]
        if y_predicted != y_true:
            error = error + 1
        w, theta = update_rule(y_true, y_predicted, X_train[i], w, theta=theta)

    return w, theta


def test(X_test, y_test, w, theta):
    '''Uses the precomputed w and theta and
     tests the performance on the test data'''

    total_error = 0
    for x, y_true in zip(X_test, y_test):
        y_predicted = perceptron(x, w, activation_func, theta)

        err = abs(y_predicted - y_true)
        total_error = total_error + err
        print x, ":", y_predicted, ":", y_true, ":", err

    avg_error = float(total_error) / num_test
    print "Total error :", total_error
    print "average error", ":", round(avg_error, 4)
    print "epsilon : ", epsilon
    if avg_error <= epsilon:
        print "Training success"
    else:
        print "Training fail"

if __name__ == "__main__":

    activation_name = sys.argv[1]
    update_name = sys.argv[2]
    ground_file = sys.argv[3]
    distribution = sys.argv[4]
    num_train = int(sys.argv[5])
    num_test = int(sys.argv[6])
    epsilon = float(sys.argv[7])
    theta = 0
    update_algos = {"winnow": winnow_update_rule,
                    "perceptron": perceptron_update_rule}
    update_rule = update_algos[update_name]
    activation_func = activation[activation_name]

    with open(ground_file) as f:
        lines = f.read().split("\n")
        name = lines[0]

        if name == "NBF":

            sign_indexes = list(map(int, re.findall("[+|-]?\d+", lines[1])))
            funs = re.findall("\w\w+", lines[1])

            n = max(map(abs, sign_indexes))

            X_train = [boolean_distribution_sample(n)
                       for i in range(num_train)]
            y_train = compute_ground_nbf(X_train, sign_indexes, funs)

            X_test = [boolean_distribution_sample(n) for i in range(num_test)]
            y_test = compute_ground_nbf(X_test, sign_indexes, funs)

        elif name == "TF":
            threshold = int(lines[1])
            coeffs = map(int, lines[2].split(" "))

            n = len(coeffs)

            if distribution == 'bool':
                X_train = [boolean_distribution_sample(n)
                           for i in range(num_train)]
                y_train = compute_ground_tf(X_train, coeffs, threshold)

                X_test = [boolean_distribution_sample(n)
                          for i in range(num_test)]
                y_test = compute_ground_tf(X_test, coeffs, threshold)

            elif distribution == 'sphere':
                X_train = [unit_sphere_distribution_sample(n)
                           for i in range(num_train)]
                y_train = compute_ground_tf(X_train, coeffs, threshold)

                X_test = [unit_sphere_distribution_sample(n)
                          for i in range(num_test)]
                y_test = compute_ground_tf(X_test, coeffs, threshold)

        else:
            print "NOT PARSEABLE"

        w, theta = train(X_train, y_train)

        test(X_test, y_test, w, theta)
