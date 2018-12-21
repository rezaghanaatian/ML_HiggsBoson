import random
import numpy as np


def split_data(x, y, ratio):
    """split the data set based on the split ratio."""
    # set seed
    seed = random.randint(1, len(y))
    np.random.seed(seed)
    # generate random indices
    num_row = len(y)
    indices = np.random.permutation(num_row)
    index_split = int(np.floor(ratio * num_row))
    index_tr = indices[: index_split]
    index_te = indices[index_split:]

    x = np.array(x)
    y = np.array(y)

    # create split
    x_tr = x[index_tr]
    x_te = x[index_te]
    y_tr = y[index_tr]
    y_te = y[index_te]
    return x_tr, x_te, y_tr, y_te


def split_data(x, ratio):
    """split the data set based on the split ratio."""
    # set seed
    seed = random.randint(1, len(x))
    np.random.seed(seed)
    # generate random indices
    num_row = len(x)
    indices = np.random.permutation(num_row)
    index_split = int(np.floor(ratio * num_row))
    index_tr = indices[: index_split]
    index_te = indices[index_split:]

    x = np.array(x)

    # create split
    x_tr = x[index_tr]
    x_te = x[index_te]
    return x_tr, x_te


def validate(y_predicted, y):
    """Compare predicted labels with correct labels and returns accuracy score!"""
    if len(y_predicted) != len(y):
        raise Exception('size of input arrays should be the same!')

    errors_count = 0
    for i in range(0, len(y)):
        if y[i] != y_predicted[i]:
            errors_count += 1

    return (len(y) - errors_count) / len(y)


# Split a dataset into k folds

def cross_validation_split(input_data, yb, n_folds=10):
    # to_do: data shuffle

    input_data_list = list()
    yb_list = list()
    input_data = list(input_data)
    yb = list(yb)
    fold_size = int(len(input_data) / n_folds)
    for i in range(n_folds):
        x_fold = list()
        yb_fold = list()
        while len(x_fold) < fold_size:
            index = np.randint(len(input_data))
            x_fold.append(input_data.pop(index))
            yb_fold.append(yb_list.pop(index))

        input_data_list.append(x_fold)
        yb_list.append(yb_fold)

    input_data_fold = np.asarray(input_data_list)
    yb_fold = np.asarray(yb_list)

    return input_data_fold, yb_fold
