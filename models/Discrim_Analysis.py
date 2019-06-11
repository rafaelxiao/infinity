import numpy as np

# Todo: in case if no X in a specific class of y, mu = xbar

class Discriminant_Analysis:
    '''
    X: (n, d)
    y: (n, c)
    types:
        - linear: LDA
        - quadratic: QDA
        - RDA: regularized LDA
        - diag: diagonal QDA, same as naive bayes
        - shrunkenCentroids: diagonal LDA with L1 shrinkage on offsets
    params:
        - type
        - N_classes
        - lambda (if RDA or shrunkenCentroids)
        - prior
        - mu ((d, c) for feature d, class c
        - (for cov)
            - sigma for QDA
            - sigma_pooled for LDA
            - beta for RDA
            - sigma_diag for diag
            - sigma_pooled_diag for shrunkenCentroids
    '''

    def __init__(self, type_of_clf):
        self.type = type_of_clf

    def fit(self, X, y, lam=None):
        if type(y) == list:
            y = np.array(y)
        self.N_classes = np.unique(y)
        if self.type in ['linear', 'lda']:
            self._fit_linear(X, y)
        elif self.type in ['quadratic', 'qda']:
            self._fit_quadratic(X, y)
        elif self.type == 'diag':
            self._fit_diag(X, y)
        elif self.type == 'rda':
            self._fit_rda(X, y, lam)
        elif self.type == 'shrunkencentroids':
            self._fit_shrunken_centroids(X, y, lam)
        else:
            pass

    def _found_ns(self, X, y):
        N_classes = np.unique(y)
        k = N_classes.shape[0]
        n, d = X.shape
        return k, d, n

    def _fit_quadratic(self, X, y):
        # What we need? mu, prior, cov
        k, d, _ = self._found_ns(X, y)
        self.prior = np.zeros(k)
        self.mu = np.zeros((d, k))
        self.cov = np.zeros((d, d, k))
        for (i, j) in enumerate(self.N_classes):
            idx = np.where(y==j)
            self.prior[i] = y[idx].shape[0]
            self.mu[:, i] = np.mean(X[idx], axis=0)
            self.cov[:, :, i] = np.cov(X, rowvar=False)
        self.prior = self.prior / np.sum(self.prior)

    def _fit_linear(self, X, y):
        k, d, n = self._found_ns(X, y)
        self.prior = np.zeros(k)
        self.mu = np.zeros((d, k))
        self.cov = np.zeros((d, d))
        for (i, j) in enumerate(self.N_classes):
            idx = np.where(y==j)
            nc = len(idx)
            self.prior[i] = y[idx].shape[0]
            self.mu[:, i] = np.mean(X[idx], axis=0)
            self.cov += nc*np.cov(X, rowvar=False)
        self.cov = self.cov / n
        self.prior = self.prior / np.sum(self.prior)

    def _fit_diag(self, X, y):
        k, d, n  = self._found_ns(X, y)
        self.prior = np.zeros(k)
        self.mu = np.zeros((d, k))
        self.cov = np.zeros(d)
        for (i, j) in enumerate(self.N_classes):
            idx = np.where(y==j)
            self.prior[i] = y[idx].shape[0]
            self.mu[:, i] = np.mean(X[idx], axis=0)
            self.cov[i] = np.var(X[idx])
        self.prior = self.prior / np.sum(self.prior)

    def _fit_rda(self, X, y, lam):
        U, D, V = np.linalg.svd(X)
        Z = U[:D.shape[0], :D.shape[0]] @ np.diag(D)
        k, d, n = self._found_ns(X, y)
        Z_cov = np.cov(Z)
        Z_cov_inv = np.linalg.inv(lam*np.diag(np.diag(Z_cov)) + (1-lam)*Z_cov)
        self.beta = np.zeros((d, k))
        for (i, j) in enumerate(self.N_classes):
            idx = np.where(y==j)
            mu = np.mean(X[idx], axis=0)
            self.beta[:, i] = V.T @ Z_cov_inv @ mu

    def _fit_shrunken_centroids(self, X, y, lam):
        k, d, n = self._found_ns(X, y)
        N_cls = np.zeros(k)
        xbar = np.mean(X, axis=0) # The centroid of whole set
        sse = np.zeros(d)
        self.mu = np.zeros((d, k))
        for (i, j) in enumerate(self.N_classes):
            idx = np.where(y==j)
            self.mu[:, i] = np.mean(X[idx], axis=0)
            N_cls[i] = len(idx[0])
            if N_cls[i] == 0:
                centroid = xbar
            else:
                centroid = np.mean(X[idx], axis=0)
            sse += np.sum(
                (X[idx] - np.repeat(centroid, N_cls[i], axis=0).reshape(-1, d))**2,
                axis=0
            )
        sigma = np.sqrt(sse / (n-k))
        s0 = np.median(sigma) # what is this?
        m = np.zeros(k) # what is this?
        offset = np.zeros((k, d))

        for (i, j) in enumerate(self.N_classes):
            if N_cls[i] == 0:
                m[i] = 0
            else:
                m[i] = np.sqrt(1/(N_cls[i] - 1/n))
            offset[i, :] = (self.mu[:, i] - xbar) / (m[i] * (sigma+s0))
            offset[i, :] = self._soft_threshold(offset[i, :], lam)
        print(m)
        print(offset)



    def _soft_threshold(self, x, delta):
        return np.sign(x) * np.max([np.abs(x)-delta, 0])

from sklearn.datasets import load_iris
iris = load_iris()
X = iris['data']
y = iris['target']

DA = Discriminant_Analysis('shrunkencentroids')
#DA.fit(X, y)
DA.fit(X, y, 0.8)

#print(DA.beta)
#print(DA.mu)
#print(DA.cov)
#print(DA.prior)
#print(DA.N_classes)
