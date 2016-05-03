__author__ = 'jatwood'

import sys
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression

import data
import kernel


def baseline_edge_experiment(model_fn, data_fn, data_name, model_name):
    print 'Running edge experiment (%s)...' % (data_name,)

    A, B, _, X, Y = data_fn()

    selection_indices = np.arange(B.shape[0])
    np.random.shuffle(selection_indices)
    selection_indices = selection_indices[:10000]

    print selection_indices

    B = B[selection_indices,:]
    X = X[selection_indices,:]
    Y = Y[selection_indices,:]

    n_edges = B.shape[0]

    indices = np.arange(n_edges)
    np.random.shuffle(indices)

    print indices

    train_indices = indices[:n_edges // 3]
    valid_indices = indices[n_edges // 3:(2* n_edges) // 3]
    test_indices  = indices[(2* n_edges) // 3:]

    best_C = None
    best_acc = float('-inf')

    for C in [10**(-x) for x in range(-4,4)]:
        m = model_fn(C)
        m.fit(X[train_indices,:], np.argmax(Y[train_indices,:],1))

        preds = m.predict(X[valid_indices,:])
        actuals = np.argmax(Y[valid_indices,:],1)

        accuracy = accuracy_score(actuals, preds)
        if accuracy > best_acc:
            best_C = C
            best_acc = accuracy

    m = model_fn(best_C)
    m.fit(X[train_indices,:], np.argmax(Y[train_indices],1))

    preds   = m.predict(X[test_indices,:])
    actuals = np.argmax(Y[test_indices,:],1)

    accuracy = accuracy_score(actuals, preds)
    f1_micro = f1_score(actuals, preds, average='micro')
    f1_macro = f1_score(actuals, preds, average='macro')

    print 'form: name,micro_f,macro_f,accuracy'
    print '###RESULTS###: %s,%s,%.8f,%.8f,%.8f' % (data_name, model_name, f1_micro, f1_macro, accuracy)

if __name__ == '__main__':
    np.random.seed()

    args = sys.argv[1:]

    name_to_data = {
        'wikirfa': lambda: data.parse_wikirfa(n_features=250)
    }

    baseline_models = {
        'logisticl1': lambda C: LogisticRegression(penalty='l1', C=C),
        'logisticl2': lambda C: LogisticRegression(penalty='l2', C=C),
    }

    data_name = args[0]
    data_fn = name_to_data[data_name]

    model_name = args[1]

    if model_name in baseline_models:
        baseline_edge_experiment(baseline_models[model_name], data_fn, data_name, model_name)
    else:
        print '%s not recognized' % (model_name,)
