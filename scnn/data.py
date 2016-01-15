__author__ = 'jatwood'

import numpy as np
import cPickle as cp


def parse_cora(plot=False):
    path = "data/cora/"

    id2index = {}

    label2index = {
        'Case_Based': 0,
        'Genetic_Algorithms': 1,
        'Neural_Networks': 2,
        'Probabilistic_Methods': 3,
        'Reinforcement_Learning': 4,
        'Rule_Learning': 5,
        'Theory': 6
    }

    features = []
    labels = []

    with open(path + 'cora.content', 'r') as f:
        i = 0
        for line in f.xreadlines():
            items = line.strip().split('\t')

            id = items[0]

            # 1-hot encode labels
            label = np.zeros(len(label2index))
            label[label2index[items[-1]]] = 1
            labels.append(label)

            # parse features
            features.append([int(x) for x in items[1:-1]])

            id2index[id] = i
            i += 1

    features = np.asarray(features, dtype='float32')
    labels = np.asarray(labels, dtype='int32')

    n_papers = len(id2index)

    adj = np.zeros((n_papers, n_papers), dtype='float32')

    with open(path + 'cora.cites', 'r') as f:
        for line in f.xreadlines():
            items = line.strip().split('\t')
            adj[ id2index[items[0]], id2index[items[1]] ] = 1.0
            # undirected
            adj[ id2index[items[1]], id2index[items[0]] ] = 1.0

    if plot:
        import networkx as nx
        import matplotlib.pyplot as plt

        #G = nx.from_numpy_matrix(adj, nx.DiGraph())
        G = nx.from_numpy_matrix(adj, nx.Graph())
        print G.order()
        print G.size()
        plt.figure()
        nx.draw(G, node_size=10, edge_size=1)
        plt.savefig('../data/cora_net.pdf')

        plt.figure()
        plt.imshow(adj)
        plt.savefig('../data/cora_adj.pdf')

    return adj.astype('float32'), features.astype('float32'), labels.astype('int32')


def parse_nci(graph_name='nci1.graph'):
    path = "data/nci/"

    with open(path+graph_name,'r') as f:
        raw = cp.load(f)

        n_classes = 2
        n_graphs = len(raw['graph'])

        A = []
        rX = []
        Y = np.zeros((n_graphs, n_classes), dtype='int32')

        for i in range(n_graphs):
            # Set label
            Y[i][raw['labels'][i]] = 1

            # Parse graph
            G = raw['graph'][i]

            n_nodes = len(G)

            a = np.zeros((n_nodes,n_nodes), dtype='float32')
            x = np.zeros((n_nodes,1), dtype='int32')

            for node, meta in G.iteritems():
                x[node,0] = meta['label'][0]
                for neighbor in meta['neighbors']:
                    a[node, neighbor] = 1

            A.append(a)
            rX.append(x)

        maxval = max([max(x) for x in rX])

        X = [np.zeros((rx.size, maxval+1), dtype='float32') for rx in rX]
        for i in range(len(X)):
            X[i][np.arange(rX[i].size),rX[i]] = 1

    return A, X, Y

