import unittest
import numpy as np
if __name__ == '__main__':
    from scipy.stats import multivariate_normal, norm

def gauss_log_prob(X, mu, sigma):

    X = np.array(X)

    # univariate
    if type(mu) == float or type(mu) == int:
        logp = -0.5 * np.log((2*sigma*np.pi)) - 0.5*(1/sigma)*(X-mu)**2
        return logp

    # multivariate single case
    else:
        mu = np.array(mu)
        X = X.reshape(-1, mu.shape[0])
        d = len(mu)
        sigma = np.array(sigma)
        norm = -d/2 * np.log(2*np.pi) - 0.5*np.log(np.linalg.det(sigma))
        logp = np.array([i for i in map(
            lambda x: norm - 0.5 * (x-mu).T @ np.linalg.inv(sigma) @ (x-mu), X)])
        if len(logp) == 1:
            logp = logp[0]
        return logp

class Test(unittest.TestCase):

    def test_gauss_log_prob(self):
        x = 6
        mean = 4.5
        sigma = 3
        r = norm.pdf(x, mean, np.sqrt(sigma))
        self.assertAlmostEqual(np.exp(gauss_log_prob(x, mean, sigma)), r, 2)

        x = [2, 1]
        mean = [1, 1]
        sigma = [[1, 0], [1, 3]]
        r = multivariate_normal.pdf(x, mean, sigma)
        self.assertAlmostEqual(np.exp(gauss_log_prob(x, mean, sigma)), r, 2)

        x = [[2, 1], [3, 3]]
        mean = [1, 1]
        sigma = [[1, 0], [1, 3]]
        r = multivariate_normal.pdf(x, mean, sigma).tolist()
        rm = np.exp(gauss_log_prob(x, mean, sigma)).tolist()
        for (i, j) in zip(rm, r):
            self.assertAlmostEqual(i, j, 2)

if __name__ == '__main__':
    unittest.main()

