__author__ = 'jatwood'

import numpy as np
import sys
from sklearn.metrics import f1_score, accuracy_score

import util, data
from graph_scnn import GraphSCNN


def graph_experiment(data_fn, name, n_hops, transform_fn=util.rw_laplacian, transform_name='rwl'):
    A, X, Y = data_fn()

    n_graphs = len(A)

    indices = np.arange(n_graphs)
    np.random.shuffle(indices)

    train_indices = indices[:n_graphs // 3]
    valid_indices = indices[n_graphs // 3:(2* n_graphs) // 3]
    test_indices  = indices[(2* n_graphs) // 3:]

    scnn = GraphSCNN(n_hops=n_hops, transform_fn=transform_fn)
    scnn.fit(A, X, Y, train_indices=train_indices, valid_indices=valid_indices,
             learning_rate=0.05, batch_size=100, n_epochs=500)

    preds = scnn.predict(A, X, test_indices)
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
        graph_experiment(lambda : data.parse_nci('nci1.graph'), 'cora', 2)
    else:
        name_to_data = {
            'nci1': lambda : data.parse_nci('nci1.graph'),
            'nci109': lambda : data.parse_nci('nci109.graph'),
            'nci1struct': lambda: data.parse_nci(graph_name='nci1.graph', with_structural_features=True),
            'nci109struct': lambda: data.parse_nci(graph_name='nci109.graph', with_structural_features=True),
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

        graph_experiment(data_fn, name, n_hops=n_hops, transform_fn=transform_fn, transform_name=transform_name)

