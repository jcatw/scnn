SCNN
====

An implementation of search-convolutional neural networks [1], a new model for graph-structured data.

Installation
------------
Using pip:

    pip install scnn

Usage
-----

	import numpy as np
    from scnn import SCNN, data
    from sklearn.metrics import f1_score

    A, X, Y = data.parse_cora()

    n_nodes = A.shape[0]

    indices = np.arange(n_nodes)
    train_indices = indices[:n_nodes // 3]
    valid_indices = indices[n_nodes // 3:(2* n_nodes) // 3]
    test_indices  = indices[(2* n_nodes) // 3:]

    scnn = SCNN()
    scnn.fit(A, X, Y, train_indices=train_indices, valid_indices=valid_indices)

    preds = scnn.predict(A, X, test_indices)
    actuals = np.argmax(Y[test_indices,:], axis=1)

    print 'F score: %.4f' % (f1_score(actuals, preds))

References
----------

[1] http://arxiv.org/abs/1511.02136
