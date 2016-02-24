__author__ = 'jatwood'

import numpy as np

import data
from scnn import SCNN

def test_scnn():
    from sklearn.metrics import f1_score

    A, X, Y = data.parse_cora()

    n_nodes = A.shape[0]

    indices = np.arange(n_nodes)
    train_indices = indices[:n_nodes // 3]
    valid_indices = indices[n_nodes // 3:(2* n_nodes) // 3]
    test_indices  = indices[(2* n_nodes) // 3:]

    scnn = SCNN(n_hops=2)
    scnn.fit(A, X, Y, train_indices=train_indices, valid_indices=valid_indices)

    probs = scnn.predict_proba(A, X, test_indices)
    print probs

    preds = scnn.predict(A, X, test_indices)
    actuals = np.argmax(Y[test_indices,:], axis=1)

    print 'F score: %.4f' % (f1_score(actuals, preds))

if __name__ == '__main__':
    import nose
    nose.run(defaultTest=__name__, argv=['.','-s'])
