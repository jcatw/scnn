__author__ = 'jatwood'

import sys
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression

import data
import util
import kernel
import structured

from baseline_graph_experiment import GraphDecompositionModel

def graph_proportion_baseline_experiment(model_fn, data_fn, data_name, model_name, prop_valid, prop_test):
    print 'Running node experiment (%s)...' % (data_name,)

    A, X, Y = data_fn()

    n_graphs = len(A)

    A = np.asarray(A)
    X = np.asarray(X)
    Y = np.asarray(Y)

    indices = np.arange(n_graphs)
    np.random.seed(4)
    np.random.shuffle(indices)
    print indices

    valid_start = int(n_graphs * (1 - (prop_valid + prop_test)))
    test_start = int(n_graphs * (1 - prop_test))

    valid_indices = indices[valid_start:test_start]
    test_indices  = indices[test_start:]

    for train_prop in [x / 10.0 for x in range(1, 11)]:
        train_end = int(valid_start * train_prop)
        train_indices = indices[:train_end]


        m = model_fn()
        m.fit_with_validation(A, X, Y, train_indices, valid_indices)

        preds   = m.predict(A, X, test_indices)
        actuals = Y[test_indices,:]

        accuracy = accuracy_score(actuals, preds)
        f1_micro = f1_score(actuals, preds, average='micro')
        f1_macro = f1_score(actuals, preds, average='macro')

        print 'form: name,micro_f,macro_f,accuracy'
        print '###RESULTS###: %s,%s,%.6f,%.8f,%.8f,%.8f' % (data_name, model_name, train_prop, f1_micro, f1_macro, accuracy)

def graph_proportion_kernel_experiment(model, data_fn, data_name, model_name, prop_valid, prop_test):
    print 'Running graph experiment (%s)...' % (data_name,)

    print 'parsing data...'

    A, X, Y = data_fn()

    print 'done'

    n_graphs = len(A)

    A = np.asarray(A)
    X = np.asarray(X)
    Y = np.asarray(Y)

    indices = np.arange(n_graphs)
    np.random.seed(4)
    np.random.shuffle(indices)
    print indices

    valid_start = int(n_graphs * (1 - (prop_valid + prop_test)))
    test_start = int(n_graphs * (1 - prop_test))

    valid_indices = indices[valid_start:test_start]
    test_indices  = indices[test_start:]

    for train_prop in [x / 10.0 for x in range(1, 11)[::-1]]:
        print 'train prop %s' % (train_prop,)
        train_end = int(valid_start * train_prop)
        train_indices = indices[:train_end]

        model.fit_with_validation(Y, train_indices, valid_indices, test_indices)

        preds = model.predict(Y, valid_indices, test_indices)
        actuals = Y[test_indices,:]

        accuracy = accuracy_score(actuals, preds)
        f1_micro = f1_score(actuals, preds, average='micro')
        f1_macro = f1_score(actuals, preds, average='macro')

        print 'form: name,micro_f,macro_f,accuracy'
        print '###RESULTS###: %s,%s,%.6f,%.8f,%.8f,%.8f' % (data_name, model_name, train_prop, f1_micro, f1_macro, accuracy)


if __name__ == '__main__':
    np.random.seed()

    args = sys.argv[1:]

    name_to_data = {
        'nci1': lambda: data.parse_nci(graph_name='nci1.graph'),
        'nci109': lambda: data.parse_nci(graph_name='nci109.graph'),
        'mutag': lambda : data.parse_nci(graph_name='mutag.graph'),
        'ptc': lambda : data.parse_nci(graph_name='ptc.graph'),
        'enzymes': lambda : data.parse_nci(graph_name='enzymes.graph'),
        'nci1struct': lambda: data.parse_nci(graph_name='nci1.graph', with_structural_features=True),
        'nci109struct': lambda: data.parse_nci(graph_name='nci109.graph', with_structural_features=True),
    }

    transform_lookup = {
        'id': None,
        'rwl': util.rw_laplacian,
        'l': util.laplacian,
    }

    name_to_parameters = {
        'nci1': {'num_dimensions':2,
                 'kernel_type':1,
                 'feature_type':3,
                 'ds_name':'nci1',
                 'window_size':2,
                 'ngram_type':0,
                 'sampling_type':1,
                 'graphlet_size':0,
                 'sample_size':2
                 },
        'nci109': {'num_dimensions':5,
                 'kernel_type':1,
                 'feature_type':3,
                 'ds_name':'nci109',
                 'window_size':10,
                 'ngram_type':0,
                 'sampling_type':0,
                 'graphlet_size':0,
                 'sample_size':2
                 },
        'mutag': {'num_dimensions':2,
                 'kernel_type':1,
                 'feature_type':3,
                 'ds_name':'mutag',
                 'window_size':2,
                 'ngram_type':0,
                 'sampling_type':1,
                 'graphlet_size':0,
                 'sample_size':2
                 },
        'enzymes': {'num_dimensions':2,
                 'kernel_type':1,
                 'feature_type':3,
                 'ds_name':'enzymes',
                 'window_size':2,
                 'ngram_type':0,
                 'sampling_type':1,
                 'graphlet_size':0,
                 'sample_size':2
                 },
        'ptc': {'num_dimensions':2,
                 'kernel_type':1,
                 'feature_type':3,
                 'ds_name':'ptc',
                 'window_size':2,
                 'ngram_type':0,
                 'sampling_type':1,
                 'graphlet_size':0,
                 'sample_size':2
                 },
        'nci1struct': {'num_dimensions':2,
                 'kernel_type':1,
                 'feature_type':3,
                 'ds_name':'nci1',
                 'window_size':2,
                 'ngram_type':0,
                 'sampling_type':1,
                 'graphlet_size':0,
                 'sample_size':2
                 },
        'nci109struct': {'num_dimensions':5,
                 'kernel_type':1,
                 'feature_type':3,
                 'ds_name':'nci109',
                 'window_size':10,
                 'ngram_type':0,
                 'sampling_type':0,
                 'graphlet_size':0,
                 'sample_size':2
                 },
    }


    data_name = args[0]
    data_fn = name_to_data[data_name]

    model_name = args[1]

    baseline_models = {
        'logisticl1': lambda: GraphDecompositionModel(reg='l1'),
        'logisticl2': lambda: GraphDecompositionModel(reg='l2')
    }

    kernel_models = {
        'deepwl': kernel.DeepWL(**name_to_parameters[data_name]),
    }


    data_name = args[0]
    data_fn = name_to_data[data_name]

    model_name = args[1]

    if model_name in baseline_models:
        graph_proportion_baseline_experiment(baseline_models[model_name], data_fn, data_name, model_name, 0.1, 0.1)
    elif model_name in kernel_models:
        graph_proportion_kernel_experiment(kernel_models[model_name], data_fn, data_name, model_name, 0.1, 0.1)
    else:
        print '%s not recognized' % (model_name,)
