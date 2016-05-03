__author__ = 'jatwood'

import numpy as np


def rw_laplacian(A):
    Dm1 = np.zeros(A.shape)

    degree = A.sum(0)

    for i in range(A.shape[0]):
        if degree[i] == 0:
            Dm1[i,i] = 0.
        else:
            Dm1[i,i] = - 1. / degree[i]

    return -np.asarray(Dm1.dot(A),dtype='float32')

def laplacian(A):
    D = np.zeros(A.shape)
    out_degree = A.sum(0)
    for i in range(A.shape[0]):
        D[i,i] = out_degree[i]

    return np.asarray(D - A, dtype='float32')

def A_power_series(A,k):
    """
    Computes [A**0, A**1, ..., A**k]

    :param A: 2d numpy array
    :param k: integer, degree of series
    :return: 3d numpy array [A**0, A**1, ..., A**k]
    """
    assert k >= 0

    Apow = [np.identity(A.shape[0])]

    if k > 0:
        Apow.append(A)

        for i in range(2, k+1):
            Apow.append(np.dot(A, Apow[-1]))

    return np.asarray(Apow, dtype='float32')



