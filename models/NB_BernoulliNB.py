import numpy as np


class BernoulliNB:
    '''
    What are the input
    - x in (N, D)
    - y in (C, 1)
    - pseudo count in (C, 1)

    What params we need to fit to make prediction?
    - prior, pi in (C, 1)
    - theta, a matrix in (C, D)

    How to do it?
    - iter through class
        - count the n of class, update prior
        - subset the x
        - iter through D:
        - count whether an elements equals 1 and sum
        - update theta (considering pseudo count)
    - normalize prior

    How to predict?
    - given a x in (N, D)
    - find positive and negative examples, x_pos, x_neg, in (N, D)
    - x_pos dot with ln(theta.T), x_neg dot with ln(1-theta.T), get two array in (N, C)
    - exp(x_pos + x_neg + ln(prior))
    - normalize within each row

    '''

    def fit(self, x, y, pseudo_count=1):
        x = np.array(x)
        y = np.array(y)
        prior = np.zeros(np.unique(y).shape[0])
        theta = np.zeros((prior.shape[0], x.shape[1]))
        for c in range(prior.shape[0]):
            idx = np.where(y == c)
            prior[c] = y[idx[0]].shape[0]
            x_sub = x[idx[0]]
            pos_s = np.sum(x_sub == 1, axis=0)
            neg_s = np.sum(x_sub == 0, axis=0)
            theta[c] = (pos_s + pseudo_count) / (pos_s + neg_s + 2 * pseudo_count)
        self.theta = theta
        self.prior = (prior / np.sum(prior))

    def predict(self, x):
        x_pos = (x == 1).astype(int)
        x_neg = (x == 0).astype(int)
        x_pos_ln = x_pos @ np.log(self.theta.T)
        x_neg_ln = x_neg @ np.log(1 - self.theta.T)
        x_pred = np.exp(x_neg_ln + x_pos_ln + np.log(self.prior))
        prob = x_pred / np.sum(x_pred, axis=1).reshape(-1, 1)
        return np.argmax(prob, axis=1), prob