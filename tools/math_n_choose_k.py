import unittest
import numpy as np
from functools import reduce


# Return the value of n choose k
def n_choose_k(n, k):

    def nck(n, k):
        r = min(k, n-k)
        numer = reduce((lambda a,b: a*b), range(n, n-r, -1))
        denom = reduce((lambda a,b: a*b), range(1, r+1, 1))
        return numer / denom

    if type(n) == list or type(n) == np.ndarray:
        n = np.array(n)
    if type(k) == list or type(k) == np.ndarray:
        k = np.array(k)

    if type(n) != int:
        return np.array([nck(a, b) for (a, b) in zip(n, k)])
    else:
        return nck(n, k)


class Test(unittest.TestCase):

    def test_n_choose_k(self):
        self.assertEqual(n_choose_k(10, 2), 45)

        n = [30, 65, 16]
        k = [3, 6, 8]
        r = [4060, 82598880, 12870]
        self.assertListEqual(
            np.round(n_choose_k(n, k)).tolist(), r
        )



if __name__ == '__main__':
    unittest.main()