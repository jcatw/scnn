__author__ = 'jatwood'

import numpy as np
from sklearn.metrics import f1_score

import data
from scnn import SCNN

def maker_fn(data_fn, name):
    print 'Testing %s...' % (name,)

    A, X, Y = data_fn()

    n_nodes = A.shape[0]

    indices = np.arange(n_nodes)
    train_indices = indices[:n_nodes // 3]
    valid_indices = indices[n_nodes // 3:(2* n_nodes) // 3]
    test_indices  = indices[(2* n_nodes) // 3:]

    scnn = SCNN(n_hops=2)
    scnn.fit(A, X, Y, train_indices=train_indices, valid_indices=valid_indices)

    probs = scnn.predict_proba(X, test_indices)
    print probs

    preds = scnn.predict(X, test_indices)
    actuals = np.argmax(Y[test_indices,:], axis=1)

    print 'F score: %.4f' % (f1_score(actuals, preds))
    print ''

def test_scnn_cora():
    maker_fn(data.parse_cora,'cora')

def test_scnn_pubmed():
    maker_fn(data.parse_pubmed,'pubmed')


if __name__ == '__main__':
    import nose
    nose.run(defaultTest=__name__, argv=['.','-s'])
