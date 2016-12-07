"""
An implementation of kernels on graphs as a baseline.
"""
__author__ = 'jatwood'

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
import util
from dgk import deep_kernel
from math import factorial
from scipy.linalg import expm

class KernelOnGraph:
    def fit(self, *args, **kwargs):
        raise NotImplementedError

    def fit_with_validation(self, *args, **kwargs):
        raise NotImplementedError

    def compute_similarities(self, Y):
        return np.dot(self.K, Y)

    def predict(self, Y, hidden_indices, prediction_indices):
        """
        Hidden indices are not visible during prediction but not predicted.
        Prediction indices are not visible during prediction and are predicted.
        """
        # copy Y and zero out validation and test labels
        #cY = np.array(Y)
        #cY[hidden_indices,:] = 0
        #cY[prediction_indices,:] = 0

        # compute similarities
        #label_similarities = self.compute_similarities(cY)

        # compute predictions (most similar label)
        #predicted_labels = np.argmax(label_similarities[prediction_indices],1)
        # 1-hot-encode the predicted labels
        #predictions = np.zeros((predicted_labels.shape[0], Y.shape[1]))
        #predictions[np.arange(predicted_labels.shape[0]), predicted_labels] = 1

        #return predictions

        preds = self.model.predict(self.K[prediction_indices,:])

        preds_1hot = np.zeros((preds.shape[0],Y.shape[1]))
        preds_1hot[np.arange(preds.shape[0]), preds] = 1
        return preds_1hot




class ExponentialDiffusionKernel(KernelOnGraph):
    """
    K_ED = exp(alpha * A)
    """

    def __init__(self):
        self.apow = None

    def fit(self, A, alpha=0.01, k = 5):
        if self.apow is None:
            self.apow = util.A_power_series(A, 5)
        self.K = sum([((alpha**k) * self.apow[i]) / factorial(i) for i in range(k+1)])


    def fit_with_validation(self, A, Y,
                            train_indices, valid_indices, test_indices,
                            alphas=[10**(-x) for x in range(3)]):
        best_acc = float('-inf')

        # try several values on alpha
        for alpha in alphas:
            for C in [(10**x) / Y.shape[0] for x in range(-7,7,2)]:
                #for C in [0.001]:
                self.fit(A, alpha=alpha)
                try:
                    self.model = LinearSVC(C=C)
                    self.model.fit(self.K[train_indices,:], Y[train_indices].argmax(1))
                    preds = self.predict(Y, test_indices, valid_indices)

                    acc = accuracy_score(Y[valid_indices], preds)
                    if acc > best_acc:
                        best_acc = acc
                        best_alpha = alpha
                        best_C = C
                    print ""
                    print "alpha = %s" % (alpha,)
                    print "C = %s" % (C,)
                    print "acc = %s" % (acc,)
                except ValueError:
                    print 'C value %s too small' % (C,)

        print best_C
        self.fit(A, alpha=best_alpha)

        self.model = LinearSVC(C=best_C)
        self.model.fit(self.K[train_indices,:], Y[train_indices].argmax(1))

        preds = self.predict(Y, test_indices, valid_indices)
        acc = accuracy_score(Y[valid_indices], preds)
        print 'acc: %s' % (acc,)

class LaplacianExponentialDiffusionKernel(ExponentialDiffusionKernel):
    """
    K_LED = exp(-alpha * L)
    """
    def __init__(self):
        print 'init model'
        self.lpow = None

    def fit(self, A, alpha=0.01, k = 5):
        if self.lpow is None:
            print 'computing lpow...'
            self.lpow = util.A_power_series(-util.laplacian(A), k)
            print 'done'
        print 'computing K...'
        self.K = sum([((alpha**k) * self.lpow[i]) / factorial(i) for i in range(k+1)])
        print 'done'

    '''
    def fit(self, A, alpha=0.01):
        L = util.laplacian(A)
        self.K = expm(-alpha * L)
    '''


class DeepWL(KernelOnGraph):
    def __init__(self,
                 num_dimensions=1, # any integer > 0
                 kernel_type=1, # 1 (deep, l2) or 2 (deep, M), 3 (MLE)
                 feature_type=3, # 1 (graphlet), 2 (SP), 3 (WL)
                 ds_name='nci1', # dataset name
                 window_size=1, # any integer > 0
                 ngram_type=1, # 1 (skip-gram), 0 (cbow)
                 sampling_type=1, # 1 (hierarchical sampling), 0 (negative sampling)
                 graphlet_size=1, # any integer > 0
                 sample_size=1 # any integer > 0
                 ):
        self.num_dimensions = num_dimensions
        self.kernel_type = kernel_type
        self.feature_type = feature_type
        self.ds_name = ds_name
        self.window_size = window_size
        self.ngram_type = ngram_type
        self.sampling_type = sampling_type
        self.graphlet_size = graphlet_size
        self.sample_size = sample_size



    def fit(self, Y, train_indices, valid_indices, test_indices, window_size=1, num_dimensions=1):
        self.num_dimensions = num_dimensions
        self.window_size = window_size

        self.K = deep_kernel.main(num_dimensions=self.num_dimensions,
                 kernel_type=self.kernel_type,
                 feature_type=self.feature_type,
                 ds_name=self.ds_name,
                 window_size=self.window_size,
                 ngram_type=self.ngram_type,
                 sampling_type=self.sampling_type,
                 graphlet_size=self.graphlet_size,
                 sample_size=self.sample_size)


    def fit_with_validation(self, Y, train_indices, valid_indices, test_indices):

        print Y
        print Y[train_indices].argmax(1)
        print train_indices
        best_acc = float('-inf')

        for ws in [2,5,10,25,50]:
            for nd in [2,5,10,25,50]:
        #for ws in [5]:
            #for nd in [5]:
                for C in [(10**x) / Y.shape[0] for x in range(-7,11,2)]:
                #for C in [0.001]:
                    print "ws = %s, nd = %s" % (ws,nd)
                    self.fit(Y, train_indices, valid_indices, test_indices, window_size=ws, num_dimensions=nd)
                    # copy Y and zero out validation and test labels
                    #cY = np.array(Y)
                    #cY[test_indices,:] = 0
                    #cY[valid_indices,:] = 0

                    #label_similarities = self.compute_similarities(cY)

                    #self.model = LinearSVC(C=C)
                    #self.model.fit(label_similarities[train_indices,:], Y[train_indices].argmax(1))
                    try:
                        self.model = LinearSVC(C=C)
                        self.model.fit(self.K[train_indices,:], Y[train_indices].argmax(1))
                        preds = self.predict(Y, test_indices, valid_indices)

                        acc = accuracy_score(Y[valid_indices], preds)
                        if acc >= best_acc:
                            best_acc = acc
                            best_ws = ws
                            best_nd = nd
                            best_C = C
                        print ""
                        print "C = %s" % (C,)
                        print "acc = %s" % (acc,)
                    except ValueError as e:
                        print e.message

        self.fit(Y, train_indices, valid_indices, test_indices, window_size=best_ws, num_dimensions=best_nd)
        # copy Y and zero out validation and test labels
        #cY = np.array(Y)
        #cY[test_indices,:] = 0
        #cY[valid_indices,:] = 0

        #label_similarities = self.compute_similarities(cY)

        print best_C
        self.model = LinearSVC(C=best_C)
        #self.model.fit(label_similarities[train_indices,:], Y[train_indices].argmax(1))
        self.model.fit(self.K[train_indices,:], Y[train_indices].argmax(1))
        preds = self.predict(Y, test_indices, valid_indices)
        acc = accuracy_score(Y[valid_indices], preds)
        print 'acc: %s' % (acc,)

    def predict(self, Y, hidden_indices, prediction_indices):
        K = self.K

        # copy Y and zero out validation and test labels
        #cY = np.array(Y)
        #cY[hidden_indices,:] = 0
        #cY[prediction_indices,:] = 0

        # compute similarities
        #label_similarities = self.compute_similarities(cY)
        preds = self.model.predict(K[prediction_indices,:])

        preds_1hot = np.zeros((preds.shape[0],Y.shape[1]))
        preds_1hot[np.arange(preds.shape[0]), preds] = 1
        return preds_1hot
