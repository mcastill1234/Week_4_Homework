from unittest import TestCase
from P7_Gradient_Descent_SVM import hinge, hinge_loss, svm_obj
import numpy as np


def super_simple_separable():
    X = np.array([[2, 3, 9, 12],
                  [5, 2, 6, 5]])
    y = np.array([[1, -1, 1, -1]])
    return X, y


sep_e_separator = np.array([[-0.40338351], [1.1849563]]), np.array([[-2.26910091]])


class Test(TestCase):
    def test_hinge(self):
        X, y = super_simple_separable()
        correct = np.array([[0, 2, 0, 2]])
        ans = hinge(y)
        print(ans, correct)

    def test_hinge_loss(self):
        X, y = super_simple_separable()
        th, th0 = np.array([[-0.40338351], [1.1849563]]), np.array([[-2.26910091]])
        ans = hinge_loss(X, y, th, th0)
        print(ans)

    def test_svm_obj(self):
        x_1, y_1 = super_simple_separable()
        th1, th1_0 = sep_e_separator
        ans = svm_obj(x_1, y_1, th1, th1_0, .1)
        print(ans)

    def test_d_hinge(self):
        self.fail()

    def test_d_hinge_loss_th(self):
        self.fail()

    def test_d_hinge_loss_th0(self):
        self.fail()

    def test_d_svm_obj_th(self):
        self.fail()

    def test_d_svm_obj_th0(self):
        self.fail()

    def test_svm_obj_grad(self):
        self.fail()

