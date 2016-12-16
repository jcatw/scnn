__author__ = 'jatwood'

import lasagne
import lasagne.layers as layers
import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt

import util


# This class is not user facing; it contains the Lasagne internals for the SCNN model.
class SearchConvolution(layers.MergeLayer):
    """
    A search-convolutional Lasagne layer.
    """
    def __init__(self, incomings, n_hops, n_features,
                 W=lasagne.init.Normal(0.01),
                 nonlinearity=lasagne.nonlinearities.tanh,
                 **kwargs):
        super(SearchConvolution, self).__init__(incomings, **kwargs)

        self.W = self.add_param(W, (n_hops,n_features), name='W')

        self.n_hops = n_hops
        self.n_features = n_features

        self.nonlinearity = nonlinearity

    def get_output_for(self, inputs, **kwargs):
        """
        Compute search convolution of inputs.

        :param inputs: [Apow, X]
        :return: Search convolution of inputs with shape (self.nhops, self.nfeatures)
        """

        Apow = inputs[0]
        X = inputs[1]

        def compute_output(i, w, a, x, h):
            """

            :param i: index
            :param w: weight vector (n_features,)
            :param x: feature vector (n_nodes, n_features)
            :param h: n_hops
            :param a: adjacency matrix (n_nodes, n_nodes)
            :return: output[i]
            """
            return (T.dot(a, x).transpose()) * T.addbroadcast(T.reshape(w, (w.shape[0],1)),1)

        seq_values = np.arange(self.n_hops)
        seq = theano.shared(value = seq_values, name="seq", borrow=True)

        out, _ = theano.scan(fn=compute_output,
                             non_sequences=[X, self.n_hops],
                             sequences=[seq, self.W, Apow],
                             n_steps = self.n_hops)

        return self.nonlinearity(out.transpose())

    def get_output_shape_for(self, input_shapes):
        print (input_shapes[1][0], self.n_hops, self.n_features)
        return (input_shapes[1][0], self.n_hops, self.n_features)

class DeepSearchConvolution(layers.Layer):
    """
    A search-convolutional Lasagne layer.
    """
    def __init__(self, incoming, n_hops, n_features,
                 W=lasagne.init.Normal(0.01),
                 nonlinearity=lasagne.nonlinearities.tanh,
                 **kwargs):
        super(DeepSearchConvolution, self).__init__(incoming, **kwargs)

        self.W = T.addbroadcast(self.add_param(W, (1,n_features,n_hops), name='W'),0)

        self.n_hops = n_hops
        self.n_features = n_features

        self.nonlinearity = nonlinearity

    def get_output_for(self, input, **kwargs):
        return self.nonlinearity(self.W * input)

    def get_output_shape_for(self, input_shape):
        print input_shape
        return input_shape


# This class is user-facing.  It contains a full SCNN model.
class SCNN:
    """
    The search-convolutional neural network model.
    """
    def __init__(self, n_hops=2, transform_fn=util.rw_laplacian):
        self.n_hops = n_hops
        self.transform_fn = transform_fn

        # Initialize Theano variables
        self.var_A = T.matrix('A')
        self.var_Apow = T.tensor3('Apow')
        self.var_X = T.matrix('X')
        self.var_Y = T.imatrix('Y')

    def _register_layers(self, batch_size, n_nodes, n_features, n_classes):
        self.l_in_apow = lasagne.layers.InputLayer((self.n_hops + 1, batch_size, n_nodes), input_var=self.var_Apow)
        self.l_in_x = lasagne.layers.InputLayer((n_nodes, n_features), input_var=self.var_X)
        self.l_sc = SearchConvolution([self.l_in_apow, self.l_in_x], self.n_hops + 1, n_features)
        self.l_out = layers.DenseLayer(self.l_sc, num_units=n_classes, nonlinearity=lasagne.nonlinearities.tanh)

    def _get_output_layer(self):
        return self.l_out

    def fit(self, A, X, Y, train_indices, valid_indices,
            learning_rate=0.05, batch_size=100, n_epochs=100,
            loss_fn=lasagne.objectives.multiclass_hinge_loss,
            update_fn=lasagne.updates.adagrad,
            stop_early=True,
            stop_window_size=5,
            output_weights=False,
            show_weights=False):

        # Ensure that data have the correct dimensions
        assert A.shape[0] == X.shape[0]
        assert X.shape[0] == Y.shape[0]
        assert len(Y.shape) > 1

        if self.transform_fn is not None:
            A = self.transform_fn(A)

        # Extract dimensions
        n_nodes = A.shape[0]
        n_features = X.shape[1] + 1
        n_classes = Y.shape[1]

        n_batch = n_nodes // batch_size

        # Compute the matrix power series
        Apow = util.A_power_series(A, self.n_hops)

        self.Apow = Apow

        # Add bias term to X
        X = np.hstack([X, np.ones((X.shape[0],1))]).astype('float32')

        # Create Lasagne layers
        self._register_layers(batch_size, n_nodes, n_features, n_classes)

        # Create symbolic representations of predictions, loss, parameters, and updates.
        prediction = layers.get_output(self._get_output_layer())
        loss = lasagne.objectives.aggregate(loss_fn(prediction, self.var_Y), mode='mean')
        params = lasagne.layers.get_all_params(self._get_output_layer())
        updates = update_fn(loss, params, learning_rate=learning_rate)

        # Create functions that apply the model to data and return loss
        apply_loss = theano.function([self.var_Apow, self.var_X, self.var_Y],
                                      loss, updates=updates)

        # Train the model
        print 'Training model...'
        validation_losses = []
        validation_loss_window = np.zeros(stop_window_size)
        validation_loss_window[:] = float('+inf')

        for epoch in range(n_epochs):
            train_loss = 0.0

            np.random.shuffle(train_indices)

            for batch in range(n_batch):
                start = batch * batch_size
                end = min((batch + 1) * batch_size, train_indices.shape[0])

                if start < end:
                    train_loss += apply_loss(Apow[:,train_indices[start:end],:],
                                             X,
                                             Y[train_indices[start:end],:])

            valid_loss = apply_loss(Apow[:,valid_indices,:],
                                    X,
                                    Y[valid_indices,:])

            print "Epoch %d training error: %.6f" % (epoch, train_loss)
            print "Epoch %d validation error: %.6f" % (epoch, valid_loss)

            validation_losses.append(valid_loss)

            if output_weights:
                W = layers.get_all_param_values(self.l_sc)[0]
                np.savetxt('W_%d.csv' % (epoch,), W, delimiter=',')

            if show_weights:
                W = layers.get_all_param_values(self.l_sc)[0]
                plt.imshow(W, aspect='auto', interpolation='none')
                plt.show()

            if stop_early:
                if valid_loss >= validation_loss_window.mean():
                    print 'Validation loss did not decrease. Stopping early.'
                    break
            validation_loss_window[epoch % stop_window_size] = valid_loss

    def predict(self, X, test_indices, A=None):
        if A is None:
            Apow = self.Apow
        else:
            if self.transform_fn is not None:
                A = self.transform_fn(A)
            # Compute the matrix power series
            Apow = util.A_power_series(A, self.n_hops)

        # add bias term to X
        X = np.hstack([X, np.ones((X.shape[0],1))]).astype('float32')

        # Create symbolic representation of predictions
        pred = layers.get_output(self.l_out)

        # Create a function that applies the model to data to predict a class
        pred_fn = theano.function([self.var_Apow, self.var_X], T.argmax(pred, axis=1), allow_input_downcast=True)

        # Return the predictions
        predictions = pred_fn(Apow[:,test_indices,:], X)
        return predictions

    def predict_proba(self, X, test_indices, A=None):
        if A is None:
            Apow = self.Apow
        else:
            if self.transform_fn is not None:
                A = self.transform_fn(A)
            # Compute the matrix power series
            Apow = util.A_power_series(A, self.n_hops)

        # add bias term to X
        X = np.hstack([X, np.ones((X.shape[0],1))]).astype('float32')

        # Create symbolic representation of predictions
        pred = layers.get_output(self.l_out)

        # Create a function that applies the model to data to predict a class
        pred_fn = theano.function([self.var_Apow, self.var_X], T.exp(pred) / T.exp(pred).sum(axis=1,keepdims=True), allow_input_downcast=True)

        # Return the predictions
        predictions = pred_fn(Apow[:,test_indices,:], X)
        return predictions



class DeepSCNN(SCNN):
    def __init__(self, n_hops=2, n_layers=4, transform_fn=util.rw_laplacian):
        self.n_hops = n_hops
        self.n_layers = n_layers
        self.transform_fn = transform_fn

        # Initialize Theano variables
        self.var_A = T.matrix('A')
        self.var_Apow = T.tensor3('Apow')
        self.var_X = T.matrix('X')
        self.var_Y = T.imatrix('Y')

    def _register_layers(self, batch_size, n_nodes, n_features, n_classes):
        self.l_in_apow = lasagne.layers.InputLayer((self.n_hops + 1, batch_size, n_nodes), input_var=self.var_Apow)
        self.l_in_x = lasagne.layers.InputLayer((n_nodes, n_features), input_var=self.var_X)
        self.l_sc = SearchConvolution([self.l_in_apow, self.l_in_x], self.n_hops + 1, n_features)
        self.l_deep = self.l_sc
        for i in range(self.n_layers):
            self.l_deep = DeepSearchConvolution(self.l_deep, n_hops=self.n_hops + 1, n_features=n_features)
        self.l_out = layers.DenseLayer(self.l_deep, num_units=n_classes, nonlinearity=lasagne.nonlinearities.tanh)
