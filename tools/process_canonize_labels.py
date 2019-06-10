import numpy as np
import unittest

def canonize_labels(x, support=None):
    '''
    1. construct a dict
        - if not support, construct using np.unique
        - if support, check validation, then iterate to get the dict
    2. use map to transform the data
    '''
    if type(x) == list:
        x = np.array(x)
    if support == None:
        u_vals = np.unique(x)
        dic = dict((u_vals[i], i) for i in range(len(u_vals)))
    else:
        if support[0] <= np.min(x) and support[1] >= np.max(x):
            dic = dict((i, i-support[0]) for i in range(support[0], support[1]+1))
        else:
            return None
    return np.array([dic[i] for i in x])

class Test(unittest.TestCase):

    def test_canonize_labels(self):
        y = ['apple', 'orange', 'apple']
        t = [0, 1, 0]
        self.assertListEqual(canonize_labels(y).tolist(), t)

        y = [20, 30, 27]
        t = [0, 10, 7]
        self.assertListEqual(canonize_labels(y, [20, 30]).tolist(), t)