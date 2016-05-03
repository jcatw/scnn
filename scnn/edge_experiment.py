__author__ = 'jatwood'

import sys
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

from edge_scnn import EdgeSCNN
import data
import util


def edge_experiment(data_fn, name, n_hops, transform_fn=util.rw_laplacian, transform_name='rwl'):
    print 'Running edge experiment (%s)...' % (name,)

    A, B, _, X, Y = data_fn()

    selection_indices = np.arange(B.shape[0])
    np.random.shuffle(selection_indices)
    selection_indices = selection_indices[:10000]

    print selection_indices

    B = B[selection_indices,:]
    X = X[selection_indices,:]
    Y = Y[selection_indices,:]

    n_edges = Y.shape[0]

    indices = np.arange(n_edges)
    np.random.shuffle(indices)

    print indices

    train_indices = indices[:n_edges // 3]
    valid_indices = indices[n_edges // 3:(2* n_edges) // 3]
    test_indices  = indices[(2* n_edges) // 3:]

    scnn = EdgeSCNN(n_hops=n_hops, transform_fn=transform_fn)
    #scnn.fit(A, B, X_N, X_E, Y,
    #         train_indices=train_indices, valid_indices=valid_indices,
    #         batch_size=10000,
    #        learning_rate=0.001)
    scnn.fit(A, B, X, Y,
             train_indices=train_indices, valid_indices=valid_indices,
             batch_size=100,
             n_epochs=500,
             learning_rate=0.05)

    probs = scnn.predict_proba(X, B, test_indices)
    print probs

    preds = scnn.predict(X, B, test_indices)
    actuals = np.argmax(Y[test_indices,:], axis=1)

    f1_micro = f1_score(actuals, preds, average='micro')
    f1_macro = f1_score(actuals, preds, average='macro')
    accuracy = accuracy_score(actuals, preds)

    print 'form: name,n_hops,transform_name,micro_f,macro_f,accuracy'
    print '###RESULTS###: %s,%d,%s,%.8f,%.8f,%.8f' % (name, n_hops, transform_name, f1_micro, f1_macro, accuracy)


if __name__ == '__main__':
    np.random.seed()
    args = sys.argv[1:]

    if len(args) == 0:
        edge_experiment(data.parse_cora, 'cora', 2)
    else:
        name_to_data = {
            'wikirfa': lambda: data.parse_wikirfa(n_features=250)
        }

        transform_lookup = {
            'id': None,
            'rwl': util.rw_laplacian,
            'l': util.laplacian,
        }

        name = args[0]
        data_fn = name_to_data[name]
        n_hops = int(args[1])
        transform_name = args[2]
        transform_fn = transform_lookup[transform_name]

        edge_experiment(data_fn, name, n_hops=n_hops, transform_fn=transform_fn, transform_name=transform_name)
