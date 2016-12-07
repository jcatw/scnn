__author__ = 'jatwood'

import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score

import data
import kernel


class GraphDecompositionModel:
    def __init__(self, reg):
        self.reg = reg

    def create_features(self, A, X):
        graph_features = []


        # Create a feature vector from the graph structure and node labels
        for a,x in zip(A,X):
            gf = x.mean(0).tolist()

            graph_features.append(gf)

        return np.asarray(graph_features)

    def fit(self, A, X, Y, train_indices, valid_indices, C=1.0):
        self.n_classes = Y.shape[1]
        training_features = self.create_features(A[train_indices], X[train_indices])

        self.model = LogisticRegression(penalty=self.reg, C=C)
        self.model.fit(training_features, Y[train_indices].argmax(1))

    def fit_with_validation(self, A, X, Y, train_indices, valid_indices):
        best_C = None
        best_acc = float('-inf')

        for C in [10**(-x) for x in range(-2,5)]:
            self.fit(A, X, Y, train_indices, valid_indices, C=C)

            preds = self.predict(A, X, valid_indices)
            truth = Y[valid_indices]

            acc = accuracy_score(truth, preds)
            if acc > best_acc:
                best_acc = acc
                best_C = C

        self.fit(A, X, Y, train_indices, valid_indices, C=best_C)

    def predict(self, A, X, indices):
        features = self.create_features(A[indices], X[indices])

        preds = self.model.predict(features)
        preds_1hot = np.zeros((preds.shape[0], self.n_classes))
        preds_1hot[np.arange(preds.shape[0]), preds] = 1

        return preds_1hot


def kernel_graph_experiment(model, data_fn, data_name, model_name):
    print 'Running graph experiment (%s)...' % (data_name,)

    A, X, Y = data_fn()

    n_nodes = len(A)

    indices = np.arange(n_nodes)
    np.random.shuffle(indices)

    print indices

    train_indices = indices[:n_nodes // 3]
    valid_indices = indices[n_nodes // 3:(2* n_nodes) // 3]
    test_indices  = indices[(2* n_nodes) // 3:]
    #train_indices = indices[:int(n_nodes*0.8)]
    #valid_indices = indices[int(n_nodes*0.8):int(n_nodes*0.9)]
    #test_indices = indices[int(n_nodes*0.9):]

    model.fit_with_validation(Y, train_indices, valid_indices, test_indices)

    preds = model.predict(Y, np.asarray([]), test_indices)
    actuals = Y[test_indices,:]

    accuracy = accuracy_score(actuals, preds)
    f1_micro = f1_score(actuals, preds, average='micro')
    f1_macro = f1_score(actuals, preds, average='macro')

    print 'form: name,micro_f,macro_f,accuracy'
    print '###RESULTS###: %s,%s,%.8f,%.8f,%.8f' % (data_name, model_name, f1_micro, f1_macro, accuracy)

def baseline_graph_experiment(model, data_fn, data_name, model_name):
    print 'Running graph experiment (%s)...' % (data_name,)

    A, X, Y = data_fn()

    A = np.asarray(A)
    X = np.asarray(X)
    Y = np.asarray(Y)

    n_nodes = A.shape[0]

    indices = np.arange(n_nodes)
    np.random.shuffle(indices)

    train_indices = indices[:n_nodes // 3]
    valid_indices = indices[n_nodes // 3:(2* n_nodes) // 3]
    test_indices  = indices[(2* n_nodes) // 3:]

    model.fit_with_validation(A, X, Y, train_indices, valid_indices)

    preds = model.predict(A, X, test_indices)
    actuals = Y[test_indices,:]

    accuracy = accuracy_score(actuals, preds)
    f1_micro = f1_score(actuals, preds, average='micro')
    f1_macro = f1_score(actuals, preds, average='macro')

    print 'form: name,micro_f,macro_f,accuracy'
    print '###RESULTS###: %s,%s,%.8f,%.8f,%.8f' % (data_name, model_name, f1_micro, f1_macro, accuracy)

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
        'logisticl1': GraphDecompositionModel(reg='l1'),
        'logisticl2': GraphDecompositionModel(reg='l2')
    }

    kernel_models = {
        'deepwl': kernel.DeepWL(**name_to_parameters[data_name]),
    }


    if model_name in kernel_models:
        kernel_graph_experiment(kernel_models[model_name], data_fn, data_name, model_name)
    elif model_name in baseline_models:
        baseline_graph_experiment(baseline_models[model_name], data_fn, data_name, model_name)
    else:
        print '%s not recognized' % (model_name,)