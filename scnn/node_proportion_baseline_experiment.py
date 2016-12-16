__author__ = 'jatwood'

import sys
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression

import data
import util
import kernel
import structured


def node_proportion_baseline_experiment(model_fn, data_fn, data_name, model_name, prop_valid, prop_test):
    print 'Running node experiment (%s)...' % (data_name,)

    A, X, Y = data_fn()

    n_nodes = A.shape[0]

    indices = np.arange(n_nodes)

    valid_start = int(n_nodes * (1 - (prop_valid + prop_test)))
    test_start = int(n_nodes * (1 - prop_test))

    valid_indices = indices[valid_start:test_start]
    test_indices  = indices[test_start:]

    for train_prop in [x / 10.0 for x in range(1, 11)]:
        train_end = int(valid_start * train_prop)
        train_indices = indices[:train_end]

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
        print '###RESULTS###: %s,%s,%.6f,%.8f,%.8f,%.8f' % (data_name, model_name, train_prop, f1_micro, f1_macro, accuracy)

def node_proportion_kernel_experiment(model, data_fn, data_name, model_name, prop_valid, prop_test):
    print 'Running node experiment (%s)...' % (data_name,)

    A, X, Y = data_fn()

    n_nodes = A.shape[0]

    indices = np.arange(n_nodes)

    valid_start = int(n_nodes * (1 - (prop_valid + prop_test)))
    test_start = int(n_nodes * (1 - prop_test))

    valid_indices = indices[valid_start:test_start]
    test_indices  = indices[test_start:]

    for train_prop in [x / 10.0 for x in range(1, 11)]:
        train_end = int(valid_start * train_prop)
        train_indices = indices[:train_end]

        model.fit_with_validation(A,Y, train_indices, valid_indices, test_indices)

        preds = model.predict(Y, valid_indices, test_indices)
        actuals = Y[test_indices,:]

        accuracy = accuracy_score(actuals, preds)
        f1_micro = f1_score(actuals, preds, average='micro')
        f1_macro = f1_score(actuals, preds, average='macro')

        print 'form: name,micro_f,macro_f,accuracy'
        print '###RESULTS###: %s,%s,%.6f,%.8f,%.8f,%.8f' % (data_name, model_name, train_prop, f1_micro, f1_macro, accuracy)

def node_proportion_structured_experiment(model, data_fn, data_name, model_name, prop_valid, prop_test):
    print 'Running node experiment (%s)...' % (data_name,)

    A, X, Y = data_fn()

    n_nodes = A.shape[0]

    indices = np.arange(n_nodes)

    valid_start = int(n_nodes * (1 - (prop_valid + prop_test)))
    test_start = int(n_nodes * (1 - prop_test))

    valid_indices = indices[valid_start:test_start]
    test_indices  = indices[test_start:]

    for train_prop in [x / 10.0 for x in range(1, 11)]:
        train_end = int(valid_start * train_prop)
        train_indices = indices[:train_end]

        model.fit_with_validation(A,X,Y, train_indices, valid_indices)

        preds = model.predict(A, X, test_indices)
        actuals = Y[test_indices,:]

        accuracy = accuracy_score(actuals, preds)
        f1_micro = f1_score(actuals, preds, average='micro')
        f1_macro = f1_score(actuals, preds, average='macro')

        print 'form: name,micro_f,macro_f,accuracy'
        print '###RESULTS###: %s,%s,%.6f,%.8f,%.8f,%.8f' % (data_name, model_name, train_prop, f1_micro, f1_macro, accuracy)


def node_proportion_crf_experiment(data_fn, data_name, model_name, prop_valid, prop_test):
    print 'Running node experiment (%s)...' % (data_name,)

    A, X, Y = data_fn()

    n_nodes = A.shape[0]

    indices = np.arange(n_nodes)

    valid_start = int(n_nodes * (1 - (prop_valid + prop_test)))
    test_start = int(n_nodes * (1 - prop_test))

    valid_indices = indices[valid_start:test_start]
    test_indices  = indices[test_start:]

    for train_prop in [x / 10.0 for x in range(1, 11)]:
        train_end = int(valid_start * train_prop)
        train_indices = indices[:train_end]

        model = structured.crfmodel(A,X,Y,train_indices,valid_indices,test_indices)

        model.fit_with_validation()

        preds = model.predict(test_indices)
        actuals = Y[test_indices,:].argmax(1)

        accuracy = accuracy_score(actuals, preds)
        f1_micro = f1_score(actuals, preds, average='micro')
        f1_macro = f1_score(actuals, preds, average='macro')

        print 'form: name,micro_f,macro_f,accuracy'
        print '###RESULTS###: %s,%s,%.6f,%.8f,%.8f,%.8f' % (data_name, 'crf', train_prop, f1_micro, f1_macro, accuracy)


if __name__ == '__main__':
    np.random.seed()

    args = sys.argv[1:]


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

    baseline_models = {
        'logisticl1': lambda C: LogisticRegression(penalty='l1', C=C),
        'logisticl2': lambda C: LogisticRegression(penalty='l2', C=C),
    }

    kernel_models = {
        'ked': kernel.ExponentialDiffusionKernel(),
        'kled': kernel.LaplacianExponentialDiffusionKernel(),
    }

    structured_models = {
        'crf-ssvm': structured.StructuredModel()
    }

    crf_models = {
        'crf': None
    }

    data_name = args[0]
    data_fn = name_to_data[data_name]

    model_name = args[1]

    if model_name in baseline_models:
        node_proportion_baseline_experiment(baseline_models[model_name], data_fn, data_name, model_name, 0.1, 0.1)
    elif model_name in kernel_models:
        node_proportion_kernel_experiment(kernel_models[model_name], data_fn, data_name, model_name, 0.1, 0.1)
    elif model_name in structured_models:
        node_proportion_structured_experiment(structured_models[model_name], data_fn, data_name, model_name, 0.1, 0.1)
    elif model_name in crf_models:
        node_proportion_crf_experiment(data_fn, data_name, model_name, 0.1, 0.1)
    else:
        print '%s not recognized' % (model_name,)
