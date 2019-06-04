import unittest
import numpy as np
from tools import n_choose_k

def binom_log_prob(x, n, theta):
    if type(x) == list or type(x) == np.ndarray:
        if type(n) == int:
            n = np.array([n for _ in x])
            theta = np.array([theta for _ in x])
    lnc = np.log(n_choose_k(n, x))
    lna = x * np.log(theta)
    lnb = (n-x) * np.log(1-theta)
    return lnc + lna + lnb

class Test(unittest.TestCase):

    def test_binom_log_prob(self):
        self.assertEqual(
            np.round(binom_log_prob(3, 6, 0.3), 2), -1.69
        )

        n = 9
        x = [6, 3, 5]
        theta = 0.4
        r = [-2.60, -1.38, -1.79]
        self.assertListEqual(
            np.round(binom_log_prob(x, n, theta), 2).tolist(), r
        )

        n = [16, 8, 11]
        x = np.array([4, 2, 2])
        theta = np.array([0.3, 0.6, 0.2])
        r = [-1.59, -3.19, -1.22]
        self.assertListEqual(
            np.round(binom_log_prob(x, n, theta), 2).tolist(), r
        )

if __name__ == '__main__':
    unittest.main()
