import unittest
import numpy as np
from math import gamma
from .math_n_choose_k import n_choose_k

# Return the log of gamma(a)
def gamma_ln(a):
    if type(a) == list:
        a = [gamma(i) for i in a]
    elif type(a) == np.ndarray:
        a = np.array([i for i in map(gamma, a)])
    else:
        a = gamma(a)
    return np.log(a)

# Return the log of beta(a)
def beta_ln(a, b):
    if type(a) == list or type(a) == np.ndarray:
        a = np.array(a)
    if type(b) == list or type(b) == np.ndarray:
        b = np.array(b)
    return gamma_ln(a) + gamma_ln(b) - gamma_ln(a+b)

# Return the log probability of beta binomial distribution
def beta_binom_log_prob(x, n, a, b):
    if type(x) == list or type(x) == np.ndarray:
        x = np.array(x)
        a = np.array([a for _ in x])
        b = np.array([b for _ in x])
        n = np.array([n for _ in x])

    return np.log(n_choose_k(n, x)) + beta_ln(x+a, n-x+b) - beta_ln(a, b)

# Tests
class Test(unittest.TestCase):

    def test_gamma_ln(self):
        self.assertEqual(
            np.round(gamma_ln(100), 2), 359.13
        )

        a = [22, 38, 138]
        b = [45.38, 99.33, 540.42]
        self.assertListEqual(
            np.round(gamma_ln(a), 2).tolist(), b
        )

        a = np.array([1, 2, 3])
        b = np.array([0.00, 0.00, 0.69])
        self.assertListEqual(
            np.round(gamma_ln(a), 2).tolist(), b.tolist()
        )

    def test_beta_ln(self):
        self.assertEqual(
            np.round(beta_ln(3, 4), 2), -4.09
        )

        a = [15, 22, 2]
        b = [33, 21, 89]
        c = [-30.05, -30.06, -8.99]
        self.assertListEqual(
            np.round(beta_ln(a, b), 2).tolist(), c
        )


    def test_beta_binom_log_prob(self):
        self.assertEqual(
            np.round(beta_binom_log_prob(6, 10, 1, 1), 2), -2.4
        )

        a = 2
        b = 5
        x = [5, 6, 9]
        n = 23
        r = [-2.38, -2.43, -2.74]
        self.assertListEqual(
            np.round(beta_binom_log_prob(x, n, a, b), 2).tolist(), r
        )



if __name__ == '__main__':
    unittest.main()