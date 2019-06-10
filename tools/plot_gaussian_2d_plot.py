import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

def gaussian_2d_plot(X, mu, cov, color, alpha=0.5, fig=None, linewidths=1):
    x1_max = np.max(X[:, 0])
    x1_min = np.min(X[:, 0])
    x2_max = np.max(X[:, 1])
    x2_min = np.min(X[:, 1])
    x1 = np.arange(x1_min, x1_max, (x1_max-x1_min)/100)
    x2 = np.arange(x2_min, x2_max, (x2_max-x2_min)/100)
    x, y = np.meshgrid(x1, x2)
    mvn = multivariate_normal(mu, cov)
    arr = []
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            point = (x[i, j], y[i, j])
            arr.append(mvn.pdf(point))
    p = np.array(arr).reshape(x.shape)
    if fig != None:
        fig.contour(x, y, p, alpha=alpha, colors=color, linewidths=linewidths)
    else:
        plt.contour(x, y, p, alpha=alpha, colors=color, linewidths=linewidths)