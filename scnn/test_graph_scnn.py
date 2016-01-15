__author__ = 'jatwood'

import numpy as np
import lasagne

import data
from graph_scnn import GraphSCNN

def test_graph_scnn():
    from sklearn.metrics import accuracy_score
    A, X, Y = data.parse_nci()

    n_graphs = len(A)

    indices = np.arange(n_graphs)
    np.random.shuffle(indices)

    train_indices = indices[:n_graphs // 3]
    valid_indices = indices[n_graphs // 3:(2* n_graphs) // 3]
    test_indices  = indices[(2* n_graphs) // 3:]

    scnn = GraphSCNN(n_hops=5)
    scnn.fit(A, X, Y, train_indices=train_indices, valid_indices=valid_indices,
             learning_rate=0.05, batch_size=100, n_epochs=500)


    preds = scnn.predict(A, X, test_indices)
    actuals = np.argmax(Y[test_indices,:], axis=1)

    print 'Accuracy: %.4f' % (accuracy_score(actuals, preds))

if __name__ == '__main__':
    import nose
    nose.run(defaultTest=__name__, argv=['.','-s'])
