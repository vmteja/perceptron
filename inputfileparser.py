import re
from operator import mul

import distribution


def compute_ground_tf(X, coeffs, th):
    ''' This computes the ground truth or the y_true,
     using the TF function, for a given input'''

    return [int(sum(map(mul, x, coeffs)) >= th) for x in X]


def funapply(num1, num2, fun):
    '''applies fun (operation : AND / OR) between num1 and num2 '''

    if fun == 'OR':
        return num1 or num2
    elif fun == 'AND':
        return num1 and num2


def compute_ground_nbf(X, sign_indexes, funs):
    ''' This computes the ground truth or the y_true,
     using the NBF function, for a given input'''

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


def nbf(lines, num_train, num_test):
    sign_indexes = list(map(int, re.findall("[+|-]?\d+", lines[1])))
    funs = re.findall("\w\w+", lines[1])

    n = max(map(abs, sign_indexes))

    X_train = [distribution.ALLOWED_DISTRIBUTIONS['bool'](n)
               for i in range(num_train)]
    y_train = compute_ground_nbf(X_train, sign_indexes, funs)

    X_test = [distribution.ALLOWED_DISTRIBUTIONS['bool'](n) for i in range(num_test)]
    y_test = compute_ground_nbf(X_test, sign_indexes, funs)
    return X_train, y_train, X_test, y_test


def tf(lines, distribution, num_train, num_test):
    threshold = int(lines[1])
    coeffs = map(int, lines[2].split(" "))

    n = len(coeffs)
    X_train = [distribution(n)
               for i in range(num_train)]
    y_train = compute_ground_tf(X_train, coeffs, threshold)

    X_test = [distribution(n)
              for i in range(num_test)]
    y_test = compute_ground_tf(X_test, coeffs, threshold)
    return X_train, y_train, X_test, y_test


def parse(ground_file, distribution, num_train, num_test):
    with open(ground_file) as f:
        lines = f.read().split("\n")
        name = lines[0]

        if name == "NBF":
            return nbf(lines, num_train, num_test)
        elif name == "TF":
            return tf(lines, distribution, num_train, num_test)
        else:
            print "NOT PARSABLE"