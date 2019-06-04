import unittest
import numpy as np
from math import gamma

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

# Return the log probability of beta distribution
def beta_log_prob(x, a, b):
    if type(x) == list or type(x) == np.ndarray:
        x = np.array(x)
        a = np.array([a for _ in x])
        b = np.array([b for _ in x])

    alnx = (a-1) * np.log(x)
    blnx = (b-1) * np.log(1-x)
    if type(alnx) == np.ndarray:
        alnx = np.array(
            [0 if a[i]==1 and x[i]==0 else alnx[i] for i in range(len(a))])
        blnx = np.array(
            [0 if b[i]==1 and x[i]==1 else blnx[i] for i in range(len(a))])
    else:
        if a==1 and x==0:
            alnx = 0
        if b==1 and x==1:
            blnx = 0
    logp = alnx + blnx - beta_ln(a, b)

    return logp


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

    def test_beta_log_prob(self):
        self.assertEqual(
            np.round(beta_log_prob(0.3, 15, 33), 2), 1.78
        )

        a = 14
        b = 56
        x = [0.3, 0.89, 0.13]
        p = [0.04, -87.60, 1.13]
        self.assertListEqual(
            np.round(beta_log_prob(x, a, b), 2).tolist(), p
        )





if __name__ == '__main__':
    unittest.main()