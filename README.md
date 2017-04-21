SCNN
====

Note: A newer, cleaner implementation is now available at https://github.com/jcatw/dcnn.

An implementation of ~~search-~~diffusion-convolutional neural networks, a new model for graph-structured data.

Installation
------------
Using pip:

    pip install scnn

Usage
-----

	import numpy as np
    from scnn import SCNN, data
    from sklearn.metrics import f1_score

	# Parse the cora dataset and return an adjacency matrix, a design matrix, and a 1-hot label matrix
    A, X, Y = data.parse_cora()

	# Construct array indices for the training, validation, and test sets
    n_nodes = A.shape[0]
    indices = np.arange(n_nodes)
    train_indices = indices[:n_nodes // 3]
    valid_indices = indices[n_nodes // 3:(2* n_nodes) // 3]
    test_indices  = indices[(2* n_nodes) // 3:]

	# Instantiate an SCNN and fit it to cora
    scnn = SCNN()
    scnn.fit(A, X, Y, train_indices=train_indices, valid_indices=valid_indices)

	# Predict labels for the test set 
    preds = scnn.predict(X, test_indices)
    actuals = np.argmax(Y[test_indices,:], axis=1)

	# Display performance
    print 'F score: %.4f' % (f1_score(actuals, preds))

What's with the S?
------------------
Historical reasons - these were once called search-convolutional neural networks.

References
----------

[1] http://arxiv.org/abs/1511.02136
