import numpy as np

class Discriminant_Analysis:
    '''
    x: (n, d)
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