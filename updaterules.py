

def perceptron_update_rule(y_true, y_predicted, x, w, theta):
    '''Takes in true and predicted target and
    returns updated weights and threshold '''

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
    '''Takes in true and predicted target and
     returns updated weights and threshold '''

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


AVAILABLE_UPDATE_RULES = {"winnow": winnow_update_rule,
                "perceptron": perceptron_update_rule}
