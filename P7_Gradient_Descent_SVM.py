import numpy as np
from P6_Gradient_Descent import num_grad


# Problem 7.1
def hinge(v):
    """
    :param v: data vector
    :return: hinge of data vector
    """
    return np.where(v >= 1, 0, 1 - v)


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


# Problem 7.2
def d_hinge(v):
    """Returns the gradient of hinge(v) with respect to v """
    return np.where(v >= 1, 0, -1)


def d_hinge_loss_th(x, y, th, th0):
    """Returns the gradient of hinge_loss(x, y, th, th0) with respect to th"""
    return d_hinge(y * (np.dot(th.T, x) + th0)) * y * x


def d_hinge_loss_th0(x, y, th, th0):
    """Returns the gradient of hinge_loss(x, y, th, th0) with respect to th0"""
    return d_hinge(y * (np.dot(th.T, x) + th0)) * y


def d_svm_obj_th(x, y, th, th0, lam):
    """Returns the gradient of svm_obj(x, y, th, th0) with respect to th"""
    d_hinge_losses = d_hinge_loss_th(x, y, th, th0)
    regular = 2*lam*th
    return np.mean(d_hinge_losses, axis=1, keepdims=True) + regular


def d_svm_obj_th0(x, y, th, th0, lam):
    """Returns the gradient of svm_obj(x, y, th, th0) with respect to th0"""
    d_hinge_losses = d_hinge_loss_th0(x, y, th, th0)
    return np.mean(d_hinge_losses, axis=1, keepdims=True)


def svm_obj_grad(X, y, th, th0, lam):
    """Returns the full gradient as a single vector (which includes both th, th0)"""
    vector_th = d_svm_obj_th(X, y, th, th0, lam)
    vector_th0 = d_svm_obj_th0(X, y, th, th0, lam)
    return np.vstack([vector_th, vector_th0])


# Test case 2
X1 = np.array([[1, 2, 3, 9, 10]])
y1 = np.array([[1, 1, 1, -1, -1]])
th1, th10 = np.array([[-0.31202807]]), np.array([[1.834]])
X2 = np.array([[2, 3, 9, 12],
               [5, 2, 6, 5]])
y2 = np.array([[1, -1, 1, -1]])
th2, th20 = np.array([[-3., 15.]]).T, np.array([[2.]])

d_hinge(np.array([[ 71.]])).tolist()
d_hinge(np.array([[ -23.]])).tolist()
d_hinge(np.array([[ 71, -23.]])).tolist()

d_hinge_loss_th(X2[:,0:1], y2[:,0:1], th2, th20).tolist()
d_hinge_loss_th(X2, y2, th2, th20).tolist()
d_hinge_loss_th0(X2[:,0:1], y2[:,0:1], th2, th20).tolist()
d_hinge_loss_th0(X2, y2, th2, th20).tolist()

d_svm_obj_th(X2[:,0:1], y2[:,0:1], th2, th20, 0.01).tolist()
d_svm_obj_th(X2, y2, th2, th20, 0.01).tolist()
d_svm_obj_th0(X2[:,0:1], y2[:,0:1], th2, th20, 0.01).tolist()
d_svm_obj_th0(X2, y2, th2, th20, 0.01).tolist()

svm_obj_grad(X2, y2, th2, th20, 0.01).tolist()
svm_obj_grad(X2[:,0:1], y2[:,0:1], th2, th20, 0.01).tolist()
