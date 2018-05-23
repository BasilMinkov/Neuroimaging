import numpy as np
import scipy


def csp(numerator, denominator, regularization=True):
    """
    def csp(numerator, denominator, regularization=True)

    Overview:

    Common spatial pattern (CSP) is a mathematical procedure used in signal processing for separating a multivariate
    signal into additive subcomponents which have maximum differences in variance between two windows [1].

    Reference:

    1. Zoltan J. Koles, Michael S. Lazaret and Steven Z. Zhou, "Spatial patterns underlying population differences in
    the background EEG", Brain topography, Vol. 2 (4) pp. 275-284, 1990
    2. https://en.wikipedia.org/wiki/Common_spatial_pattern

    :param numerator: CSP numerator
    :param denominator: CSP denominator
    :param regularization: enable or disable Tikhonov regularization
    :param n_channels: number of channels or components
    :return:
    """

    # Calculate covariance matrices
    covariance_real = np.dot(numerator.T, numerator) / numerator.shape[0]
    covariance_mock = np.dot(denominator.T, denominator) / denominator.shape[0]

    if regularization:

        C1 = covariance_real / covariance_real.diagonal().sum()
        C2 = covariance_mock / covariance_mock.diagonal().sum()

        # Tikhonov regularization
        covariance_real = C1 + 0.5 * C1.diagonal().sum() * np.identity(C1.shape[1]) / C1.shape[0]
        covariance_mock = C2 + 0.5 * C2.diagonal().sum() * np.identity(C2.shape[1]) / C2.shape[0]

    # Find eigenvalues and eigenvectors
    values, vectors = scipy.linalg.eigh(covariance_real, covariance_mock)

    return values, vectors