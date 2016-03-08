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

def parse_pubmed():
    path = 'data/Pubmed-Diabetes/data/'

    n_nodes = 19717
    n_features = 500
    n_classes = 3

    data_X = np.zeros((n_nodes, n_features), dtype='float32')
    data_Y = np.zeros((n_nodes, n_classes), dtype='int32')

    paper_to_index = {}
    feature_to_index = {}

    # parse nodes
    with open(path + 'Pubmed-Diabetes.NODE.paper.tab','r') as node_file:
        # first two lines are headers
        node_file.readline()
        node_file.readline()

        k = 0

        for i,line in enumerate(node_file.xreadlines()):
            items = line.strip().split('\t')

            paper_id = items[0]
            paper_to_index[paper_id] = i

            # label=[1,2,3]
            label = int(items[1].split('=')[-1]) - 1  # subtract 1 to zero-count
            data_Y[i,label] = 1.

            # f1=val1 \t f2=val2 \t ... \t fn=valn summary=...
            features = items[2:-1]
            for feature in features:
                parts = feature.split('=')
                fname = parts[0]
                fvalue = float(parts[1])

                if fname not in feature_to_index:
                    feature_to_index[fname] = k
                    k += 1

                data_X[i, feature_to_index[fname]] = fvalue

    # parse graph
    data_A = np.zeros((n_nodes, n_nodes), dtype='float32')

    with open(path + 'Pubmed-Diabetes.DIRECTED.cites.tab','r') as edge_file:
        # first two lines are headers
        edge_file.readline()
        edge_file.readline()

        for i,line in enumerate(edge_file.xreadlines()):

            # edge_id \t paper:tail \t | \t paper:head
            items = line.strip().split('\t')

            edge_id = items[0]

            tail = items[1].split(':')[-1]
            head = items[3].split(':')[-1]

            data_A[paper_to_index[tail],paper_to_index[head]] = 1.0
            data_A[paper_to_index[head],paper_to_index[tail]] = 1.0

    return data_A, data_X, data_Y


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

