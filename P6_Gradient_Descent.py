import numpy as np


def rv(value_list):
    return np.array([value_list])


def cv(value_list):
    return np.transpose(rv(value_list))


def f1(x):
    return float((2 * x + 3) ** 2)


def df1(x):
    return float(2 * 2 * (2 * x + 3))


def f2(v):
    x = float(v[0]);
    y = float(v[1])
    return (x - 2.) * (x - 3.) * (x + 3.) * (x + 1.) + (x + y - 1) ** 2


def df2(v):
    x = float(v[0]);
    y = float(v[1])
    return cv([(-3. + x) * (-2. + x) * (1. + x) + \
               (-3. + x) * (-2. + x) * (3. + x) + \
               (-3. + x) * (1. + x) * (3. + x) + \
               (-2. + x) * (1. + x) * (3. + x) + \
               2 * (-1. + x + y),
               2 * (-1. + x + y)])


def gd(f, df, x0, step_size_fn, max_iter):
    """
    Finds the value of x that minimizes the objective function f using gradient descent.
    :param f: a function whose input is x, a column vector, and returns a scalar.
    :param df: a function whose input is x, a column vector, and returns a column vector representing
    the gradient of f at x.
    :param x0: an initial value of x, which is a column vector
    :param step_size_fn: a function that is given the iteration index (an integer) and returns a step size.
    :param max_iter: the number of iterations to perform
    :return: a tuple (x, fs, xs) where x is the value at the final step, fs is the list of values of f found
    during all the iterations (including f(x0)) and xs is the list of values of x found during all iterations
    (including x0).
    """
    prev_x = x0
    fs = []
    xs = []
    for i in range(max_iter):
        prev_f, prev_grad = f(prev_x), df(prev_x)
        fs.append(prev_f)
        xs.append(prev_x)
        if i == max_iter - 1:
            return prev_x, fs, xs
        step = step_size_fn(i)
        prev_x = prev_x - step * prev_grad


def package_ans(gd_vals):
    x, fs, xs = gd_vals
    return [x.tolist(), [fs[0], fs[-1]], [xs[0].tolist(), xs[-1].tolist()]]


# Test cases for gd function.
# Test case 1
# ans = package_ans(gd(f1, df1, cv([0.]), lambda i: 0.1, 1000))
# print(ans)

# Test case 2
# ans = package_ans(gd(f2, df2, cv([0., 0.]), lambda i: 0.01, 1000))
# print(ans)


def num_grad(f, delta=0.001):
    """
    Takes objective function f and a value of delta and returns a new function that takes a column vector x and
    returns a gradient column vector.
    :param f: objective function
    :param delta: step size.
    :return: function df providing the estimated gradient by means of finite differences method.
    """
    def df(x):
        g = np.zeros(x.shape)
        for i in range(x.shape[0]):
            xi = x[i, 0]
            x[i, 0] = xi - delta
            fxm = f(x)
            x[i, 0] = xi + delta
            fxp = f(x)
            x[i, 0] = xi
            g[i, 0] = (fxp - fxm) / (2 * delta)
        return g
    return df


# Test cases for num_grad function
# x = cv([0.])
# ans = (num_grad(f1)(x).tolist(), x.tolist())
# print(ans)
#
# x = cv([0.1])
# ans = (num_grad(f1)(x).tolist(), x.tolist())
# print(ans)
#
# x = cv([0., 0.])
# ans = (num_grad(f2)(x).tolist(), x.tolist())
# print(ans)
#
# x = cv([0.1, -0.1])
# ans = (num_grad(f2)(x).tolist(), x.tolist())
# print(ans)


def minimize(f, x0, step_size_fn, max_iter):
    """
    Takes a function f and uses numerical gradient descent to return the local minimum.
    :param f: objective function
    :param x0: an initial value of x, which is a column vector
    :param step_size_fn: a function that is given the iteration index (an integer) and returns a step size.
    :param max_iter: the number of iterations to perform
    :return: function f's local minimum
    """
    df = num_grad(f)
    return gd(f, df, x0, step_size_fn, max_iter)


# Test cases for minimize function
# ans = package_ans(minimize(f1, cv([0.]), lambda i: 0.1, 1000))
# print(ans)
# ans = package_ans(minimize(f2, cv([0., 0.]), lambda i: 0.01, 1000))
# print(ans)