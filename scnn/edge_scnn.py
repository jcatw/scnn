__author__ = 'jatwood'

import lasagne
import lasagne.layers as layers
import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt

from scnn import SearchConvolution
import util

# This class is not user facing; it contains the Lasagne internals for the SCNN model.
class EdgeSearchConvolution(layers.MergeLayer):
    """
    A search-convolutional Lasagne layer.
    """
    def __init__(self, incomings, n_hops, n_edge_features,
                 W=lasagne.init.Normal(0.01),
                 nonlinearity=lasagne.nonlinearities.tanh,
                 **kwargs):
        super(EdgeSearchConvolution, self).__init__(incomings, **kwargs)

        self.W = self.add_param(W, (n_hops, n_edge_features), name='W')

        self.n_hops = n_hops
        self.n_edge_features = n_edge_features

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
        return (input_shapes[0][0], self.n_hops, self.n_edge_features)


# This class is user-facing.  It contains a full EdgeSCNN model.
class EdgeSCNN:
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

    def _convert_A(self, B, A):
        n_nodes = A.shape[0]
        n_edges = B.shape[0]

        # new A: [[A, B^T], [B, 0]]
        new_A = np.zeros((n_nodes + n_edges, n_nodes + n_edges), dtype='float32')

        new_A[:n_nodes,:n_nodes] = A
        new_A[:n_nodes, n_nodes:] = B.transpose()
        new_A[n_nodes:,:n_nodes] = B

        return new_A

    def _convert_X(self, B, X):
        n_nodes = B.shape[1]
        n_edges = X.shape[0]
        n_features = X.shape[1]

        # new X: [n_nodes zeros, X]
        new_X = np.zeros((n_nodes + n_edges, n_features), dtype='float32')
        new_X[n_nodes:,:] = X

        return new_X

    def fit(self, A, B, X, Y, train_indices, valid_indices,
            learning_rate=0.05, batch_size=100, n_epochs=100,
            loss_fn=lasagne.objectives.multiclass_hinge_loss,
            update_fn=lasagne.updates.adagrad,
            stop_early=True,
            stop_window_size=5,
            output_weights=False,
            show_weights=False):

        # Ensure that data have the correct dimensions
        assert B.shape[0] == X.shape[0]
        assert B.shape[0] == Y.shape[0]
        assert len(Y.shape) > 1

        # Add bias term to X
        X = np.hstack([X, np.ones((X.shape[0],1))]).astype('float32')

        print 'correct A and X'
        A = self._convert_A(B, A)
        X = self._convert_X(B, X)

        if self.transform_fn is not None:
            A = self.transform_fn(A)

        # Extract dimensions
        n_nodes = B.shape[1]
        n_edges = B.shape[0]
        n_edge_features = X.shape[1]
        n_classes = Y.shape[1]

        n_batch = n_edges // batch_size

        print 'Compute the matrix power series'
        Apow = util.A_power_series(A, self.n_hops)
        print Apow.shape
        self.Apow = Apow

        print 'Create Lasagne layers'
        self.l_in_apow = lasagne.layers.InputLayer((self.n_hops + 1, n_nodes+n_edges, n_nodes+n_edges), input_var=self.var_Apow)
        self.l_in_x = lasagne.layers.InputLayer((n_edges, n_edge_features), input_var=self.var_X)
        self.l_sc = EdgeSearchConvolution([self.l_in_apow, self.l_in_x], self.n_hops + 1, n_edge_features)
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

            corrected_train_indices = train_indices + n_nodes

            for batch in range(n_batch):
                start = batch * batch_size
                end = min((batch + 1) * batch_size, train_indices.shape[0])

                if start < end:
                    train_loss += apply_loss(Apow[:,corrected_train_indices[start:end],:], X, Y[train_indices[start:end],:])

            corrected_valid_indices = valid_indices + n_nodes
            valid_loss = apply_loss(Apow[:,corrected_valid_indices,:], X, Y[valid_indices,:])

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

    def predict(self, X, B, test_indices, A=None,):
        if A is None:
            Apow = self.Apow
        else:
            A = self._convert_A(B, A)
            if self.transform_fn is not None:
                A = self.transform_fn(A)
            # Compute the matrix power series
            Apow = util.A_power_series(A, self.n_hops)

        n_nodes = B.shape[1]

        corrected_test_indices = test_indices + n_nodes

        # add bias term to X
        X= np.hstack([X, np.ones((X.shape[0],1))]).astype('float32')

        X = self._convert_X(B, X)

        # Create symbolic representation of predictions
        pred = layers.get_output(self.l_out)

        # Create a function that applies the model to data to predict a class
        pred_fn = theano.function([self.var_Apow, self.var_X], T.argmax(pred, axis=1), allow_input_downcast=True)

        # Return the predictions
        predictions = pred_fn(Apow[:,corrected_test_indices,:], X)
        return predictions

    def predict_proba(self, X, B, test_indices, A=None):
        if A is None:
            Apow = self.Apow
        else:
            A = self._convert_A(B, A)
            if self.transform_fn is not None:
                A = self.transform_fn(A)
            # Compute the matrix power series
            Apow = util.A_power_series(A, self.n_hops)

        n_nodes = B.shape[1]

        corrected_test_indices = test_indices + n_nodes

        # add bias term to X
        X= np.hstack([X, np.ones((X.shape[0],1))]).astype('float32')

        X = self._convert_X(B, X)

        # Create symbolic representation of predictions
        pred = layers.get_output(self.l_out)

        # Create a function that applies the model to data to predict a class
        pred_fn = theano.function([self.var_Apow, self.var_X], T.exp(pred) / T.exp(pred).sum(axis=1,keepdims=True), allow_input_downcast=True)

        # Return the predictions
        predictions = pred_fn(Apow[:,corrected_test_indices,:], X)
        return predictions

# This class is not user facing; it contains the Lasagne internals for the SCNN model.
class IncidenceEdgeSearchConvolution(layers.MergeLayer):
    """
    A search-convolutional Lasagne layer.
    """
    def __init__(self, incomings, n_hops, n_node_features, n_edge_features,
                 W_N=lasagne.init.Normal(0.01),
                 W_E=lasagne.init.Normal(0.01),
                 nonlinearity=lasagne.nonlinearities.tanh,
                 **kwargs):
        super(IncidenceEdgeSearchConvolution, self).__init__(incomings, **kwargs)

        self.W_N = self.add_param(W_N, (n_hops, n_node_features), name='W_N')
        self.W_E = self.add_param(W_E, (n_hops, n_edge_features), name='W_E')

        self.n_hops = n_hops
        self.n_node_features = n_node_features
        self.n_edge_features = n_edge_features

        self.nonlinearity = nonlinearity

    def get_output_for(self, inputs, **kwargs):
        """
        Compute search convolution of inputs.

        :param inputs: [Apow, X]
        :return: Search convolution of inputs with shape (self.nhops, self.nfeatures)
        """

        B_left = inputs[0]
        Apow = inputs[1]
        B = inputs[2]
        X_N = inputs[3]
        X_E = inputs[4]

        # Node portion
        def scan_n_hops_nodes(i, w, a, bl, b, x):
            return T.dot(bl,T.dot(a,x)) * w

        # Edge portion
        def scan_n_hops_edges(i, w, a, bl, b, x):
            return T.dot(bl,T.dot(a,T.dot(T.transpose(b),x))) * w

        hop_seq_values = np.arange(self.n_hops)
        hop_seq = theano.shared(value = hop_seq_values, name="hop_seq", borrow=True)

        node_output, _ = theano.scan(fn=scan_n_hops_nodes,
                             sequences=[hop_seq, self.W_N, Apow],
                             non_sequences=[B_left, B, X_N],
                             n_steps = self.n_hops)
        node_output = self.nonlinearity(node_output)

        edge_output, _ = theano.scan(fn=scan_n_hops_edges,
                             sequences=[hop_seq, self.W_E, Apow],
                             non_sequences=[B_left, B, X_E],
                             n_steps = self.n_hops)
        edge_output = self.nonlinearity(edge_output)

        output = T.concatenate([node_output, edge_output], axis=1)

        output = output.dimshuffle((1,0,2))

        return output

    def get_output_shape_for(self, input_shapes):
        B_left_shape = input_shapes[0]
        Apow_shape = input_shapes[1]
        B_shape = input_shapes[2]
        X_N_shape = input_shapes[3]
        X_E_shape = input_shapes[4]

        n_edges = B_left_shape[0]
        n_node_features = X_N_shape[1]
        n_edge_features = X_E_shape[1]
        n_hops = Apow_shape[0]

        #output_shape = (n_hops, n_edges, n_node_features + n_edge_features)
        output_shape = (n_edges, n_hops, n_node_features + n_edge_features)
        print output_shape
        return output_shape



# This class is user-facing.  It contains a full EdgeSCNN model.
class IncidenceEdgeSCNN:
    """
    The search-convolutional neural network model.
    """
    def __init__(self, n_hops=2, transform_fn=util.rw_laplacian):
        self.n_hops = n_hops
        self.transform_fn = transform_fn

        # Initialize Theano variables
        self.var_A = T.matrix('A')
        self.var_Apow = T.tensor3('Apow')
        self.var_B = T.matrix('B')
        self.var_B_left = T.matrix('B_left')
        self.var_X_N = T.matrix('X_N')
        self.var_X_E = T.matrix('X_E')
        self.var_Y = T.imatrix('Y')

    def fit(self, A, B, X_N, X_E, Y, train_indices, valid_indices,
            learning_rate=0.05, batch_size=100, n_epochs=100,
            loss_fn=lasagne.objectives.multiclass_hinge_loss,
            update_fn=lasagne.updates.adagrad,
            stop_early=True,
            stop_window_size=5,
            output_weights=False,
            show_weights=False):

        # Ensure that data have the correct dimensions
        assert A.shape[0] == X_N.shape[0]
        assert B.shape[1] == X_N.shape[0]
        assert B.shape[0] == X_E.shape[0]
        assert B.shape[0] == Y.shape[0]
        assert len(Y.shape) > 1

        if self.transform_fn is not None:
            A = self.transform_fn(A)

        # Extract dimensions
        n_nodes = X_N.shape[0]
        n_edges = B.shape[0]
        n_node_features = X_N.shape[1] + 1
        n_edge_features = X_E.shape[1]
        n_classes = Y.shape[1]

        n_batch = n_nodes // batch_size

        # Compute the matrix power series
        #Apow = np.asarray([util.A_power_series(A[i], self.n_hops) for i in range(n_edge_features)])
        Apow = util.A_power_series(A, self.n_hops)
        print Apow.shape

        self.Apow = Apow

        # Add bias term to X
        X_N = np.hstack([X_N, np.ones((X_N.shape[0],1))]).astype('float32')

        # Create Lasagne layers
        self.l_in_b_left = lasagne.layers.InputLayer((batch_size, n_nodes), input_var=self.var_B_left)
        self.l_in_apow = lasagne.layers.InputLayer((self.n_hops + 1, n_nodes, n_nodes), input_var=self.var_Apow)
        self.l_in_b = lasagne.layers.InputLayer((n_edges, n_nodes), input_var=self.var_B)
        self.l_in_x_n = lasagne.layers.InputLayer((n_nodes, n_node_features), input_var=self.var_X_N)
        self.l_in_x_e = lasagne.layers.InputLayer((n_edges, n_edge_features), input_var=self.var_X_E)
        self.l_sc = IncidenceEdgeSearchConvolution([self.l_in_b_left, self.l_in_apow, self.l_in_b, self.l_in_x_n, self.l_in_x_e], self.n_hops + 1, n_node_features, n_edge_features)
        self.l_out = layers.DenseLayer(self.l_sc, num_units=n_classes, nonlinearity=lasagne.nonlinearities.tanh)

        # Create symbolic representations of predictions, loss, parameters, and updates.
        prediction = layers.get_output(self.l_out)
        loss = lasagne.objectives.aggregate(loss_fn(prediction, self.var_Y), mode='mean')
        params = lasagne.layers.get_all_params(self.l_out)
        updates = update_fn(loss, params, learning_rate=learning_rate)

        # Create functions that apply the model to data and return loss
        apply_loss = theano.function([self.var_B_left, self.var_Apow, self.var_B, self.var_X_N, self.var_X_E, self.var_Y],
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
                    train_loss += apply_loss(B[train_indices[start:end],:],
                        Apow,
                                             B,
                                             X_N,
                                             X_E,
                                             Y[train_indices[start:end],:])

            valid_loss = apply_loss(B[valid_indices,:],
                Apow,
                                    B,
                                    X_N,
                                    X_E,
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

    def predict(self, X_N, X_E, B, test_indices, A=None):
        if A is None:
            Apow = self.Apow
        else:
            if self.transform_fn is not None:
                A = self.transform_fn(A)
            # Compute the matrix power series
            Apow = util.A_power_series(A, self.n_hops)

        # add bias term to X
        X_N = np.hstack([X_N, np.ones((X_N.shape[0],1))]).astype('float32')

        # Create symbolic representation of predictions
        pred = layers.get_output(self.l_out)

        # Create a function that applies the model to data to predict a class
        pred_fn = theano.function([self.var_B_left, self.var_Apow, self.var_B, self.var_X_N, self.var_X_E], T.argmax(pred, axis=1), allow_input_downcast=True)

        # Return the predictions
        predictions = pred_fn(B[test_indices,:], Apow, B, X_N, X_E)
        return predictions

    def predict_proba(self, X_N, X_E, B, test_indices, A=None):
        if A is None:
            Apow = self.Apow
        else:
            if self.transform_fn is not None:
                A = self.transform_fn(A)
            # Compute the matrix power series
            Apow = util.A_power_series(A, self.n_hops)

        # add bias term to X
        X = np.hstack([X_N, np.ones((X_N.shape[0],1))]).astype('float32')

        # Create symbolic representation of predictions
        pred = layers.get_output(self.l_out)

        # Create a function that applies the model to data to predict a class
        pred_fn = theano.function([self.var_B_left, self.var_Apow, self.var_B, self.var_X_N, self.var_X_E],
                                  T.exp(pred) / T.exp(pred).sum(axis=1,keepdims=True),
                                  allow_input_downcast=True)

        # Return the predictions
        predictions = pred_fn(B[test_indices,:], Apow, B, X_N, X_E)
        return predictions


# This class is not user facing; it contains the Lasagne internals for the SCNN model.
class EdgesAreSearchConvolution(layers.MergeLayer):
    """
    A search-convolutional Lasagne layer.
    """
    def __init__(self, incomings, n_hops, n_node_features, n_edge_features,
                 W=lasagne.init.Normal(0.01),
                 nonlinearity=lasagne.nonlinearities.tanh,
                 **kwargs):
        super(EdgesAreSearchConvolution, self).__init__(incomings, **kwargs)

        self.W = self.add_param(W, (n_edge_features, n_hops, n_node_features), name='W')

        self.n_hops = n_hops
        self.n_node_features = n_node_features
        self.n_edge_features = n_edge_features

        self.nonlinearity = nonlinearity

    def get_output_for(self, inputs, **kwargs):
        """
        Compute search convolution of inputs.

        :param inputs: [Apow, X]
        :return: Search convolution of inputs with shape (self.nhops, self.nfeatures)
        """

        Apow = inputs[0]
        B = inputs[1]
        X = inputs[2]

        def scan_n_hops(i, w, a, b, x):
            """

            :param i: index
            :param w: weight vector (n_features,)
            :param x: feature vector (n_nodes, n_features)
            :param h: n_hops
            :param a: adjacency matrix (n_nodes, n_nodes)
            :return: output[i]
            """
            #output shape: n_node_features x n_edges
            return (T.dot(T.dot(b,a), x).transpose()) * T.addbroadcast(T.reshape(w, (w.shape[0],1)),1)

        def scan_edge_features(i, w, a, b, x, nhops):
            hop_seq_values = np.arange(self.n_hops)
            hop_seq = theano.shared(value = hop_seq_values, name="hop_seq", borrow=True)

            # output shape: n_hops x n_node_features x n_edges
            out, _ = theano.scan(fn=scan_n_hops,
                                 sequences=[hop_seq, w, a],
                                 non_sequences=[b, x],
                                 n_steps = nhops)

            return out

        edge_seq_values = np.arange(self.n_edge_features)
        edge_seq = theano.shared(value = edge_seq_values, name="edge_seq", borrow=True)

        # output shape: n_edge_features x n_hops x n_node_features x n_edges
        out, _ = theano.scan(fn=scan_edge_features,
                             sequences=[edge_seq, self.W, Apow],
                             non_sequences=[B, X, self.n_hops],
                             n_steps = self.n_edge_features)

        # desired output shape: n_edges x n_edge_features x n_hops x n_node_features
        out = out.dimshuffle((3, 0, 1, 2))
        return self.nonlinearity(out)

    def get_output_shape_for(self, input_shapes):
        print (input_shapes[1][0], self.n_edge_features, self.n_hops, self.n_node_features)
        return (input_shapes[1][0], self.n_edge_features, self.n_hops, self.n_node_features)

# This class is user-facing.  It contains a full EdgeSCNN model.
class EdgesAreSearchSCNN:
    """
    The search-convolutional neural network model.
    """
    def __init__(self, n_hops=2, transform_fn=util.rw_laplacian):
        self.n_hops = n_hops
        self.transform_fn = transform_fn

        # Initialize Theano variables
        self.var_A = T.matrix('A')
        self.var_Apow = T.tensor4('Apow')
        self.var_B = T.matrix('B')
        self.var_X = T.matrix('X')
        self.var_Y = T.imatrix('Y')

    def fit(self, A, B, X, Y, train_indices, valid_indices,
            learning_rate=0.05, batch_size=100, n_epochs=100,
            loss_fn=lasagne.objectives.multiclass_hinge_loss,
            update_fn=lasagne.updates.adagrad,
            stop_early=True,
            stop_window_size=5,
            output_weights=False,
            show_weights=False):

        # Ensure that data have the correct dimensions
        assert A.shape[1] == X.shape[0]
        assert B.shape[1] == X.shape[0]
        assert B.shape[0] == Y.shape[0]
        assert len(Y.shape) > 1

        if self.transform_fn is not None:
            A = np.asarray([self.transform_fn(A[i]) for i in range(A.shape[0])])

        # Extract dimensions
        n_nodes = X.shape[0]
        n_edges = B.shape[0]
        n_node_features = X.shape[1] + 1
        n_edge_features = A.shape[0]
        n_classes = Y.shape[1]

        n_batch = n_nodes // batch_size

        # Compute the matrix power series
        Apow = np.asarray([util.A_power_series(A[i], self.n_hops) for i in range(n_edge_features)])
        print Apow.shape

        self.Apow = Apow

        # Add bias term to X
        X = np.hstack([X, np.ones((X.shape[0],1))]).astype('float32')

        # Create Lasagne layers
        self.l_in_apow = lasagne.layers.InputLayer((n_edge_features, self.n_hops + 1, n_nodes, n_nodes), input_var=self.var_Apow)
        self.l_in_b = lasagne.layers.InputLayer((batch_size, n_nodes), input_var=self.var_B)
        self.l_in_x = lasagne.layers.InputLayer((n_nodes, n_node_features), input_var=self.var_X)
        self.l_sc = EdgesAreSearchConvolution([self.l_in_apow, self.l_in_b, self.l_in_x],  self.n_hops + 1, n_node_features, n_edge_features)
        self.l_out = layers.DenseLayer(self.l_sc, num_units=n_classes, nonlinearity=lasagne.nonlinearities.tanh)

        # Create symbolic representations of predictions, loss, parameters, and updates.
        prediction = layers.get_output(self.l_out)
        loss = lasagne.objectives.aggregate(loss_fn(prediction, self.var_Y), mode='mean')
        params = lasagne.layers.get_all_params(self.l_out)
        updates = update_fn(loss, params, learning_rate=learning_rate)

        # Create functions that apply the model to data and return loss
        apply_loss = theano.function([self.var_Apow, self.var_B, self.var_X, self.var_Y],
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
                    train_loss += apply_loss(Apow,
                                             B[train_indices[start:end],:],
                                             X,
                                             Y[train_indices[start:end],:])

            valid_loss = apply_loss(Apow,
                                    B[valid_indices,:],
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

    def predict(self, X, B, test_indices, A=None):
        if A is None:
            Apow = self.Apow
        else:
            if self.transform_fn is not None:
                A = np.asarray([self.transform_fn(A[i]) for i in range(A.shape[0])])
            # Compute the matrix power series
            Apow = np.asarray([util.A_power_series(A[i], self.n_hops) for i in range(A.shape[0])])

        # add bias term to X
        X = np.hstack([X, np.ones((X.shape[0],1))]).astype('float32')

        # Create symbolic representation of predictions
        pred = layers.get_output(self.l_out)

        # Create a function that applies the model to data to predict a class
        pred_fn = theano.function([self.var_Apow, self.var_B, self.var_X], T.argmax(pred, axis=1), allow_input_downcast=True)

        # Return the predictions
        predictions = pred_fn(Apow, B[test_indices,:], X)
        return predictions

    def predict_proba(self, X, B, test_indices, A=None):
        if A is None:
            Apow = self.Apow
        else:
            if self.transform_fn is not None:
                A = np.asarray([self.transform_fn(A[i]) for i in range(A.shape[0])])
            # Compute the matrix power series
            Apow = np.asarray([util.A_power_series(A[i], self.n_hops) for i in range(A.shape[0])])

        # add bias term to X
        X = np.hstack([X, np.ones((X.shape[0],1))]).astype('float32')

        # Create symbolic representation of predictions
        pred = layers.get_output(self.l_out)

        # Create a function that applies the model to data to predict a class
        pred_fn = theano.function([self.var_Apow, self.var_B, self.var_X], T.exp(pred) / T.exp(pred).sum(axis=1,keepdims=True), allow_input_downcast=True)

        # Return the predictions
        predictions = pred_fn(Apow, B[test_indices,:], X)
        return predictions
