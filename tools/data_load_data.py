import os
import pandas as pd
import scipy.io

def load_data(name):
    path = os.path.abspath(os.path.join(os.getcwd(), "..%s..%sdatasets"%(os.sep, os.sep)))
    forma = name.split('.')[-1]
    if forma == 'mat':
        return scipy.io.loadmat(os.path.join(path, name))
    if forma == 'csv':
        return pd.read_csv(os.path.join(path, name))