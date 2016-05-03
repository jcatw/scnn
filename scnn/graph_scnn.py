__author__ = 'jatwood'

import lasagne
import lasagne.layers as layers
import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt

import util

# This class is not user facing; it contains the Lasagne internals for the SCNN model.
class GraphSearchConvolution(layers.MergeLayer):
    """
    A search-convolutional Lasagne layer.
    """
    def __init__(self, incomings, n_hops, n_features,
                 op=T.mean,
                 W=lasagne.init.Normal(0.01),
                 nonlinearity=lasagne.nonlinearities.tanh,
                 **kwargs):
        super(GraphSearchConvolution, self).__init__(incomings, **kwargs)

        self.op = op

        self.n_hops = n_hops
        self.n_features = n_features

        self.W = self.add_param(W, (self.n_hops, self.n_features), name='W')

        self.nonlinearity = nonlinearity

    def get_output_for(self, inputs, **kwargs):
        """
        Compute search convolution of inputs.

        :param inputs: [Apow, X]
        :return: Search convolution of inputs with shape (self.nhops, self.nfeatures)
        """

        Apow = inputs[0]
        X = inputs[1]

        def compute_output(w, a, x):
            return self.op(T.dot(a, x).transpose()) * T.addbroadcast(T.reshape(w, (w.shape[0],1)),1)

        def scan_hops(apow, x, w, h):
            out, _ = theano.scan(fn=compute_output,
                                 non_sequences=[x],
                                 sequences=[w, apow],
                                 n_steps = h)

            return self.nonlinearity(out.transpose())

        out, _ = theano.scan(fn=scan_hops,
                             non_sequences=[self.W, self.n_hops],
                             sequences=[Apow, X],
                             n_steps = X.shape[0])

        return out

    def get_output_shape_for(self, input_shapes):
        shape = (input_shapes[0][0], self.n_hops, self.n_features)
        print shape
        return shape


# This class is user-facing.  It contains a full SCNN model.
class GraphSCNN():
    """
    The graph search-convolutional neural network model.
    """

    def __init__(self,
                 n_hops=2,
                 ops=(T.mean, T.min, T.max, T.std, T.ptp),
                 transform_fn=util.rw_laplacian):
        self.n_hops = n_hops
        self.ops = ops
        self.transform_fn = transform_fn

        # Initialize Theano variables
        self.var_A = T.tensor3('A')
        self.var_Apow = T.tensor4('Apow')
        self.var_X = T.tensor3('X')
        self.var_Y = T.imatrix('Y')

    def fit(self, A, X, Y, train_indices, valid_indices,
            learning_rate=0.05, batch_size=100, n_epochs=100,
            loss_fn=lasagne.objectives.multiclass_hinge_loss,
            update_fn=lasagne.updates.adagrad,
            stop_early=True,
            stop_window_size=5,
            output_weights=False,
            show_weights=False):
        assert len(A) == len(X)
        assert len(X) == Y.shape[0]
        assert len(Y.shape) > 1

        if self.transform_fn is not None:
            A = [self.transform_fn(a) for a in A]

        # Extract dimensions
        n_graphs= len(A)
        n_features = X[0].shape[1] + 1
        n_classes = Y.shape[1]
        max_nodes = max([a.shape[0] for a in A])

        n_batch = n_graphs // batch_size

        # Compute the matrix power series, zero-padding for graphs with fewer than max nodes
        Apow_seq = [util.A_power_series(a, self.n_hops) for a in A]
        Apow = np.zeros((n_graphs, self.n_hops + 1, max_nodes, max_nodes), dtype='float32')
        for i, apow in enumerate(Apow_seq):
            n_nodes = apow.shape[1]
            Apow[i,:,:n_nodes, :n_nodes] = apow

        # zero-pad X and add bias term
        X_pad = np.zeros((n_graphs, max_nodes, n_features), dtype='float32')
        for i in range(n_graphs):
            n_nodes = X[i].shape[0]
            X_pad[i,:n_nodes,-1] = 1
            X_pad[i,:n_nodes,:-1] = X[i]

        X = X_pad

        # Create Lasagne layers
        self.l_in_apow = lasagne.layers.InputLayer((batch_size, self.n_hops + 1, max_nodes, max_nodes), input_var=self.var_Apow)
        self.l_in_x = lasagne.layers.InputLayer((n_graphs, max_nodes, n_features), input_var=self.var_X)
        self.l_sc_components = [GraphSearchConvolution([self.l_in_apow, self.l_in_x], self.n_hops + 1, n_features, op=op) for op in self.ops]
        #self.l_sc_mean = GraphSearchConvolution([self.l_in_apow, self.l_in_x], self.n_hops + 1, n_features, op=T.mean)
        #self.l_sc_min = GraphSearchConvolution([self.l_in_apow, self.l_in_x], self.n_hops + 1, n_features, op=T.min)
        #self.l_sc_max = GraphSearchConvolution([self.l_in_apow, self.l_in_x], self.n_hops + 1, n_features, op=T.max)
        #self.l_sc = layers.ConcatLayer([self.l_sc_mean, self.l_sc_min, self.l_sc_max])
        self.l_sc = layers.ConcatLayer(self.l_sc_components)
        self.l_out = layers.DenseLayer(self.l_sc, num_units=n_classes, nonlinearity=lasagne.nonlinearities.tanh)

        # Create symbolic representations of predictions, loss, parameters, and updates.
        prediction = layers.get_output(self.l_out)
        loss = lasagne.objectives.aggregate(loss_fn(prediction, self.var_Y), mode='mean')
        params = lasagne.layers.get_all_params(self.l_out)
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
                    train_loss += apply_loss(Apow[train_indices[start:end],:,:,:],
                                             X[train_indices[start:end],:,:],
                                             Y[train_indices[start:end],:])

            valid_loss = apply_loss(Apow[valid_indices,:,:,:],
                                    X[valid_indices,:,:],
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

    def predict(self, A, X, test_indices):
        if self.transform_fn is not None:
            A = [self.transform_fn(a) for a in A]

        # Extract dimensions
        n_graphs= len(A)
        n_features = X[0].shape[1] + 1
        max_nodes = max([a.shape[0] for a in A])

        # Compute the matrix power series
        Apow_seq = [util.A_power_series(a, self.n_hops) for a in A]
        Apow = np.zeros((n_graphs, self.n_hops + 1, max_nodes, max_nodes))
        for i, apow in enumerate(Apow_seq):
            n_nodes = apow.shape[1]
            Apow[i,:,:n_nodes, :n_nodes] = apow

        X_pad = np.zeros((n_graphs, max_nodes, n_features), dtype='float32')
        for i in range(n_graphs):
            n_nodes = X[i].shape[0]
            X_pad[i,:n_nodes,:] = np.hstack([X[i], np.ones(X[i].shape[0]).reshape((X[i].shape[0],1))])  # add the bias feature in here
        X = X_pad

        # Create symbolic representation of predictions
        pred = layers.get_output(self.l_out)

        # Create a function that applies the model to data to predict a class
        pred_fn = theano.function([self.var_Apow, self.var_X], T.argmax(pred, axis=1), allow_input_downcast=True)

        # Return the predictions
        predictions = pred_fn(Apow[test_indices,:,:,:], X[test_indices,:,:])

        return predictions



