import os
import scipy.io

def load_data(name):
    path = os.path.abspath(os.path.join(os.getcwd(), "..%s..%spmtk3data"%(os.sep, os.sep)))
    forma = name.split('.')[-1]
    if forma == 'mat':
        return scipy.io.loadmat(os.path.join(path, name))