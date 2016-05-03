"""
An implementation of kernels on graphs as a baseline.
"""
__author__ = 'jatwood'

import numpy as np
from scipy.linalg import expm
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
import util
from dgk import deep_kernel

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
        cY = np.array(Y)
        cY[hidden_indices,:] = 0
        cY[prediction_indices,:] = 0

        # compute similarities
        label_similarities = self.compute_similarities(cY)

        # compute predictions (most similar label)
        predicted_labels = np.argmax(label_similarities[prediction_indices],1)
        # 1-hot-encode the predicted labels
        predictions = np.zeros((predicted_labels.shape[0], Y.shape[1]))
        predictions[np.arange(predicted_labels.shape[0]), predicted_labels] = 1

        return predictions


class ExponentialDiffusionKernel(KernelOnGraph):
    """
    K_ED = exp(alpha * A)
    """

    def fit(self, A, alpha=0.01):
        self.K = expm(alpha * A)


    def fit_with_validation(self, A, Y,
                            train_indices, valid_indices, test_indices,
                            alphas=[10**(-x) for x in range(6)]):
        best_K = None
        best_acc = float('-inf')

        # try several values on alpha
        for alpha in alphas:
            print "alpha = %s..." % (alpha,)
            self.fit(A, alpha=alpha)

            preds = self.predict(Y, test_indices, valid_indices)

            accuracy = accuracy_score(Y[valid_indices], preds)
            if accuracy > best_acc:
                best_acc = accuracy
                best_K = self.K

        # select the most accurate kernel
        self.K = best_K

class LaplacianExponentialDiffusionKernel(ExponentialDiffusionKernel):
    """
    K_LED = exp(-alpha * L)
    """
    def fit(self, A, alpha=0.01):
        L = util.laplacian(A)
        self.K = expm(-alpha * L)


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

        '''
        # copy Y and zero out validation and test labels
        cY = np.array(Y)
        cY[test_indices,:] = 0
        cY[valid_indices,:] = 0

        label_similarities = self.compute_similarities(cY)

        self.model = LinearSVC(C=C)
        self.model.fit(label_similarities[train_indices,:], Y[train_indices].argmax(1))
        '''


    def fit_with_validation(self, Y, train_indices, valid_indices, test_indices):
        best_C = None
        best_acc = float('-inf')
        best_model = None
        for ws in [2,5,10,25,50]:
            for nd in [2,5,10,25,50]:
                print "ws = %s, nd = %s" % (ws,nd)
                self.fit(Y, train_indices, valid_indices, test_indices, window_size=ws, num_dimensions=nd)
                preds = self.predict(Y, test_indices, valid_indices)

                acc = accuracy_score(Y[valid_indices], preds)
                if acc > best_acc:
                    best_acc = acc
                    best_ws = ws
                    best_nd = nd
                print "acc = %s" % (acc,)

        self.fit(Y, train_indices, valid_indices, test_indices, window_size=best_ws, num_dimensions=best_nd)

    '''
    def predict(self, Y, hidden_indices, prediction_indices):
        K = self.K

        # copy Y and zero out validation and test labels
        cY = np.array(Y)
        cY[hidden_indices,:] = 0
        cY[prediction_indices,:] = 0

        # compute similarities
        label_similarities = self.compute_similarities(cY)
        preds = self.model.predict(label_similarities[prediction_indices,:])

        preds_1hot = np.zeros((preds.shape[0],Y.shape[1]))
        preds_1hot[np.arange(preds.shape[0]), preds] = 1
        return preds_1hot
    '''
