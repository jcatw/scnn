__author__ = 'jatwood'

import sys
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

from scnn import SCNN
import data
import util


def node_experiment(data_fn, name, n_hops, transform_fn=util.rw_laplacian, transform_name='rwl'):
    print 'Running node experiment (%s)...' % (name,)

    A, X, Y = data_fn()

    n_nodes = A.shape[0]

    indices = np.arange(n_nodes)
    np.random.shuffle(indices)

    train_indices = indices[:n_nodes // 3]
    valid_indices = indices[n_nodes // 3:(2* n_nodes) // 3]
    test_indices  = indices[(2* n_nodes) // 3:]

    scnn = SCNN(n_hops=n_hops, transform_fn=transform_fn)
    scnn.fit(A, X, Y, train_indices=train_indices, valid_indices=valid_indices)

    probs = scnn.predict_proba(X, test_indices)
    print probs

    preds = scnn.predict(X, test_indices)
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
        node_experiment(data.parse_cora, 'cora', 2)
    else:
        name_to_data = {
            'cora': data.parse_cora,
            'pubmed': data.parse_pubmed,
            'blogcatalog': data.parse_blogcatalog,
        }

        transform_lookup = {
            'id': None,
            'rwl': util.rw_laplacian,
            'l': util.laplacian,
        }

        name = args[0]
        data_fn = name_to_data[name]
        n_hops = int(args[1])
        transform_name = sys.argv[3]
        transform_fn = transform_lookup[transform_name]

        node_experiment(data_fn, name, n_hops=n_hops, transform_fn=transform_fn, transform_name=transform_name)
