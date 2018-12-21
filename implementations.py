# -*- coding: utf-8 -*-
"""some ML methods."""
import numpy as np


def compute_gradient(y, tx, w):
    """Compute the gradient."""
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad, err


def calculate_mse(e):
    """Calculate the mse for vector e."""
    return 1 / 2 * np.mean(e ** 2)


def calculate_mae(e):
    """Calculate the mae for vector e."""
    return np.mean(np.abs(e))


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Returns:
        w: last weight vector of the method,
        loss: corresponding loss value(cost function)
    """
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # compute loss, gradient
        grad, err = compute_gradient(y, tx, w)
        loss = calculate_mse(err)
        # gradient w by descent update
        w -= gamma * grad
        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}".format(
            bi=n_iter, ti=max_iters - 1, l=loss))
        # print(w)
    return losses[-1], ws[-1]


def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient for batch data."""
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad, err


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


def compute_loss(y, tx, w):
    """Calculate the loss.
    You can calculate the loss using mse or mae.
    """
    e = y - tx.dot(w)
    # return calculate_mse(e)
    return calculate_mae(e)


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Stochastic gradient descent."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    batch_size = 1
    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=1):
            # compute a stochastic gradient and loss
            grad, _ = compute_stoch_gradient(y_batch, tx_batch, w)
            # update w through the stochastic gradient update
            w -= gamma * grad
            # calculate loss
            loss = compute_loss(y, tx, w)
            # store w and loss
            ws.append(w)
            losses.append(loss)

        print("SGD({bi}/{ti}): loss={l}".format(
            bi=n_iter, ti=max_iters - 1, l=loss))
        # print(w)
    return losses[-1], ws[-1]


def least_squares(y, tx):
    """calculate the least squares solution."""
    G = np.transpose(tx).dot(tx)
    Ginv = np.linalg.inv(G)
    w = Ginv.dot(np.transpose(tx)).dot(y)
    loss = compute_loss(y, tx, w)
    return loss, w


"""def ridge_regression(y, tx, lambda_):
    aI = lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    return np.linalg.solve(a, b) """


def ridge_regression(y, tx, lmbd):
    """implement ridge regression."""
    G = np.transpose(tx).dot(tx)
    N = tx.shape[0]
    D = tx.shape[1]
    inv = np.linalg.inv(G + lmbd * 2 * N * np.identity(D))
    w = inv.dot(np.transpose(tx)).dot(y)
    loss = compute_loss(y, tx, w)
    return loss, w


# /====================================== Logisitc Regressio ==========================================\ 


def sigmoid(t):
    """Sigmoid function implementation"""
#     return 1.0 / (1.0 + np.exp(-t))
    return 0.5 * (1 + np.tanh(0.5*t)) # used this to avoid potential overflow

def compute_loss_logistic_reg(y, tx, w):
    """compute cost by means of (log) likelihood"""
    epsilon = 1e-5
    pred = sigmoid(tx.dot(w))
    loss = y.T.dot(np.log(pred + epsilon)) + (1 - y).T.dot(np.log(1 - pred+ epsilon))
    return np.squeeze(-loss)


def compute_gradient_logistic_reg(y, tx, w):
    """compute loss gradient"""
    pred = sigmoid(tx.dot(w))
    gradient = tx.T.dot(pred - y)
    return gradient


def gradient_descent_logistic_reg(y, tx, w, gamma):
    """ Compute GD using logistic regression """
    loss = compute_loss_logistic_reg(y, tx, w)
    gradient = compute_gradient_logistic_reg(y, tx, w)
    w -= gamma * gradient
    return loss, w # return new w and the loss


def logistic_regression_GD(y, tx, initial_w, max_iters, gamma):
    # initialize w
    w = initial_w

    losses = []
    threshold = 1e-9

    # start interation for calculating logistic regression
    for iter in range(max_iters):
        # compute loss and update w.
        loss, w = gradient_descent_logistic_reg(y, tx, initial_w, gamma)

        # create a list of losses
        losses.append(loss)

        # check for convergance
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    return loss, w



# /=========================== Regularized Logisitc Regression ======================================\

def compute_loss_reg_logistic_reg(y,tx, w, lambda_):
    epsilon = 1e-5
    pred = sigmoid(tx.dot(w))
    loss = (-y * np.log(pred + epsilon) - (1 - y) * np.log(1 - pred + epsilon)).sum() + 2*(w.T@w)
    return loss    

def gradient_descent_reg_logistic_reg(y, tx, w, gamma, lambda_):
    """ Calculate GD by regularized regression """
    loss = compute_loss_reg_logistic_reg(y,tx, w, lambda_)
    gradient = compute_gradient_logistic_reg(y, tx, w) + 2 * lambda_ * w
    w -= gamma * gradient
    return loss, w


def reg_logistic_regression_GD(y, tx, initial_w, max_iters, gamma,lambda_):
    # initialize w
    w = initial_w

    threshold = 1e-9
    losses = []

    # start interation for calculating logistic regression
    for iter in range(max_iters):
        # compute loss and update w
        loss, w = gradient_descent_reg_logistic_reg(y, tx, w, gamma,lambda_)

        # print the loss every 50 iterations
#         if iter % 50 == 0:
#             print("Iter={}, loss={}".format(iter, loss))

        # create a list of losses
        losses.append(loss)

        # check for convergance
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

#     print("loss={loss}".format(loss=compute_loss_reg_log_reg(y, tx, w,lambda_)))
    return loss, w
    
