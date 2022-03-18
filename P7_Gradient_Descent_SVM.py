import numpy as np


def hinge(v):
    """
    :param v: data vector
    :return: hinge of data vector
    """
    return np.where(v >= 1, 0, 1 - v)


# x is dxn, y is 1xn, th is dx1, th0 is 1x1
def hinge_loss(x, y, th, th0):
    """
    Calculates the hinge loss for data set x
    :param x: data as a d x n matrix of d dimensions and n data samples.
    :param y: data labels as a 1 x n column vector
    :param th: theta as d x 1 vector with parameters for the separator
    :param th0: theta_0 scalar parameter for the separator
    :return: the hinge loss for each point in the data set
    """
    return hinge(y * (np.dot(th.T, x) + th0))


# x is dxn, y is 1xn, th is dx1, th0 is 1x1, lam is a scalar
def svm_obj(x, y, th, th0, lam):
    """
    Calculates the SVM objective function as the mean of the hinge loss over all points and introduces regularization.
    :param x: data as a d x n matrix of d dimensions and n data samples.
    :param y: data labels as a 1 x n column vector
    :param th: theta as d x 1 vector with parameters for the separator
    :param th0: theta_0 scalar parameter for the separator
    :param lam: regularizator parameter lambda
    :return: SVM objective value
    """
    return np.mean(hinge_loss(x, y, th, th0)) + lam * np.linalg.norm(th) ** 2


def super_simple_separable():
    """Simple data and labels set for testing"""
    X = np.array([[2, 3, 9, 12],
                  [5, 2, 6, 5]])
    y = np.array([[1, -1, 1, -1]])
    return X, y


# Test case 1
# sep_e_separator = np.array([[-0.40338351], [1.1849563]]), np.array([[-2.26910091]])
# x_1, y_1 = super_simple_separable()
# th1, th1_0 = sep_e_separator
# ans = svm_obj(x_1, y_1, th1, th1_0, .1)


def d_hinge(v):
    """Returns the gradient of hinge(v) with respect to v """
    return None


def d_hinge_loss_th(x, y, th, th0):
    """Returns the gradient of hinge_loss(x, y, th, th0) with respect to th"""
    return None


def d_hinge_loss_th0(x, y, th, th0):
    """Returns the gradient of hinge_loss(x, y, th, th0) with respect to th0"""
    return None


def d_svm_obj_th(x, y, th, th0, lam):
    """Returns the gradient of svm_obj(x, y, th, th0) with respect to th"""
    return None


# Returns the gradient of svm_obj(x, y, th, th0) with respect to th0
def d_svm_obj_th0(x, y, th, th0, lam):
    return None


def svm_obj_grad(X, y, th, th0, lam):
    """Returns the full gradient as a single vector (which includes both th, th0)"""
    return None
