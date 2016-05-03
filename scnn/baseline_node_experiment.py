__author__ = 'jatwood'

import sys
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression

import data
import kernel


def baseline_node_experiment(model_fn, data_fn, data_name, model_name):
    print 'Running node experiment (%s)...' % (data_name,)

    A, X, Y = data_fn()

    n_nodes = A.shape[0]

    indices = np.arange(n_nodes)
    np.random.shuffle(indices)

    train_indices = indices[:n_nodes // 3]
    valid_indices = indices[n_nodes // 3:(2* n_nodes) // 3]
    test_indices  = indices[(2* n_nodes) // 3:]

    best_C = None
    best_acc = float('-inf')

    for C in [10**(-x) for x in range(-4,4)]:
        m = model_fn(C)
        m.fit(X[train_indices,:], np.argmax(Y[train_indices,:],1))

        preds = m.predict(X[valid_indices])
        actuals = np.argmax(Y[valid_indices,:],1)

        accuracy = accuracy_score(actuals, preds)
        if accuracy > best_acc:
            best_C = C
            best_acc = accuracy

    m = model_fn(best_C)
    m.fit(X[train_indices], np.argmax(Y[train_indices],1))

    preds   = m.predict(X[test_indices])
    actuals = np.argmax(Y[test_indices,:],1)

    accuracy = accuracy_score(actuals, preds)
    f1_micro = f1_score(actuals, preds, average='micro')
    f1_macro = f1_score(actuals, preds, average='macro')

    print 'form: name,micro_f,macro_f,accuracy'
    print '###RESULTS###: %s,%s,%.8f,%.8f,%.8f' % (data_name, model_name, f1_micro, f1_macro, accuracy)

def kernel_node_experiment(model, data_fn, data_name, model_name):
    print 'Running node experiment (%s)...' % (data_name,)

    A, X, Y = data_fn()

    n_nodes = A.shape[0]

    indices = np.arange(n_nodes)
    np.random.shuffle(indices)

    train_indices = indices[:n_nodes // 3]
    valid_indices = indices[n_nodes // 3:(2* n_nodes) // 3]
    test_indices  = indices[(2* n_nodes) // 3:]

    model.fit_with_validation(A,Y, train_indices, valid_indices, test_indices)

    preds = model.predict(Y, valid_indices, test_indices)
    actuals = Y[test_indices,:]

    accuracy = accuracy_score(actuals, preds)
    f1_micro = f1_score(actuals, preds, average='micro')
    f1_macro = f1_score(actuals, preds, average='macro')

    print 'form: name,micro_f,macro_f,accuracy'
    print '###RESULTS###: %s,%s,%.8f,%.8f,%.8f' % (data_name, model_name, f1_micro, f1_macro, accuracy)

if __name__ == '__main__':
    np.random.seed()

    args = sys.argv[1:]

    if len(args) == 0:
        baseline_node_experiment(data.parse_cora, 'cora', 2)
    else:
        name_to_data = {
            'cora': data.parse_cora,
            'pubmed': data.parse_pubmed,
            'blogcatalog': data.parse_blogcatalog,
        }

        baseline_models = {
            'logisticl1': lambda C: LogisticRegression(penalty='l1', C=C),
            'logisticl2': lambda C: LogisticRegression(penalty='l2', C=C),
        }

        kernel_models = {
            'ked': kernel.ExponentialDiffusionKernel(),
            'kled': kernel.LaplacianExponentialDiffusionKernel(),
        }

        data_name = args[0]
        data_fn = name_to_data[data_name]

        model_name = args[1]

        if model_name in baseline_models:
            baseline_node_experiment(baseline_models[model_name], data_fn, data_name, model_name)
        elif model_name in kernel_models:
            kernel_node_experiment(kernel_models[model_name], data_fn, data_name, model_name)
        else:
            print '%s not recognized' % (model_name,)

