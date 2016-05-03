__author__ = 'jatwood'

import numpy as np
import cPickle as cp

import inspect
import os

current_dir = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))


def parse_cora(plot=False):
    path = "%s/data/cora/" % (current_dir,)

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
    path = '%s/data/Pubmed-Diabetes/data/' % (current_dir,)

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


def parse_nci(graph_name='nci1.graph', with_structural_features=False):
    path = "%s/data/nci/" % (current_dir,)

    if graph_name == 'nci1.graph':
        maxval = 37
    elif graph_name == 'nci109.graph':
        maxval = 38

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
            x = np.zeros((n_nodes,maxval), dtype='float32')

            for node, meta in G.iteritems():
                x[node,meta['label'][0] - 1] = 1
                for neighbor in meta['neighbors']:
                    a[node, neighbor] = 1

            A.append(a)
            rX.append(x)

    if with_structural_features:
        import networkx as nx

        for i in range(len(rX)):
            struct_feat = np.zeros((rX[i].shape[0], 3))
            # degree
            struct_feat[:,0] = A[i].sum(1)

            G = nx.from_numpy_matrix(A[i])
            # pagerank
            prank = nx.pagerank_numpy(G)
            struct_feat[:,1] = np.asarray([prank[k] for k in range(A[i].shape[0])])

            # clustering
            clust = nx.clustering(G)
            struct_feat[:,2] = np.asarray([clust[k] for k in range(A[i].shape[0])])

            rX[i] = np.hstack((rX[i],struct_feat))

    return A, rX, Y


def parse_blogcatalog(target=1):
    path = "%s/data/blogcatalog/" % (current_dir,)

    n_nodes = 10312
    n_features = 38
    n_classes = 2

    X = np.zeros((n_nodes, n_features), dtype='float32')
    Y = np.zeros((n_nodes, n_classes), dtype='int32')

    # parse features
    with open(path+'blogcatalog_features.csv','r') as f:
        for i,line in enumerate(f.xreadlines()):
            items = np.asarray([int(x) for x in line.strip().split(',')])
            X[i,:] = np.hstack([items[:target], items[target+1:]])
            Y[i,items[target]] = 1

    # parse graph
    A = np.genfromtxt(path+'blogcatalog_network.csv', delimiter=',').astype('float32')

    return A, X, Y

def parse_wikirfa(n_features = 100):
    '''Parse wiki-rfa for an edge sentiment analysis experiment.'''

    #from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import HashingVectorizer

    path = "%s/data/wikirfa/" % (current_dir,)

    n_nodes = 10926
    n_edges = 176963  # number of +/- edges

    n_edges_read = 189004  # includes neutral edges

    # username -> index map constructed on the fly
    usermap = {}

    A = np.zeros((n_nodes, n_nodes), dtype='float32')
    B = np.zeros((n_edges, n_nodes), dtype='float32')
    X_N = np.ones((n_nodes, 1) , dtype='float32')
    X_E = np.ones((n_edges, n_features) , dtype='float32')
    Y = np.zeros((n_edges, 2), dtype='int32')

    # build comment vectorizer
    with open(path+'all_comments.txt','r') as f:
        #vectorizer = CountVectorizer('content', max_features=n_features)
        vectorizer = HashingVectorizer('content', n_features=n_features)
        vectorizer.fit(f.xreadlines())

    u = 0
    v = 0

    resmap  = {-1: 0, 1: 1}
    votemap = {-1: 0, 1: 1}

    with open(path + 'wiki-RfA.txt','r') as f:
        for i in range(n_edges_read):
            # Read in the entry
            # SRC:Guettarda
            tail = f.readline().strip()[4:]

            # TGT:Lord Roem
            head = f.readline().strip()[4:]

            # VOT:1
            vote = int(f.readline().strip()[4:])

            # RES:1
            res = int(f.readline().strip()[4:])

            # YEA:2013
            year = int(f.readline().strip()[4:])

            #DAT:19:53, 25 January 2013
            date = f.readline().strip()[4:]

            # TXT:'''Support''' per [[WP:DEAL]]: clueful, and unlikely to break Wikipedia.
            txt = f.readline().strip()[4:]

            # kill blank line
            f.readline()

            # Process the entry
            # index users
            if tail not in usermap:
                usermap[tail] = u
                u += 1
            if head not in usermap:
                usermap[head] = u
                u += 1

            if vote != 0:
                # add vote edge
                A[usermap[tail], usermap[head]] = vote
                A[usermap[head], usermap[tail]] = vote

                # add to incidence matrix
                B[v, usermap[tail]] = 1
                B[v, usermap[head]] = 1

                X_N[usermap[head],0] = resmap[res]
                X_E[v, :] = np.asarray(vectorizer.transform([txt]).todense())[0]

                Y[v, votemap[vote]] = 1
                v += 1

    print u
    print v

    assert len(usermap) == n_nodes
    assert u == n_nodes
    assert v == n_edges

    return A, B, X_N, X_E, Y

def parse_wikirfa_edges_as_search(n_features = 100):
    '''Parse wiki-rfa for an edge sentiment analysis experiment.'''

    from sklearn.feature_extraction.text import HashingVectorizer

    path = "%s/data/wikirfa/" % (current_dir,)

    n_nodes = 10926
    n_edges = 176963  # number of +/- edges

    n_edges_read = 189004  # includes neutral edges

    # username -> index map constructed on the fly
    usermap = {}

    A = np.zeros((n_features + 1, n_nodes, n_nodes), dtype='float32')
    B = np.zeros((n_edges, n_nodes), dtype='float32')
    X = np.ones((n_nodes, 1) , dtype='float32')
    Y = np.zeros((n_edges, 2), dtype='int32')

    # build comment vectorizer
    with open(path+'all_comments.txt','r') as f:
        vectorizer = HashingVectorizer('content', n_features=n_features)
        vectorizer.fit(f.xreadlines())

    u = 0
    v = 0

    resmap  = {-1: 0, 1: 1}
    votemap = {-1: 0, 1: 1}

    with open(path + 'wiki-RfA.txt','r') as f:
        for i in range(n_edges_read):
            # Read in the entry
            # SRC:Guettarda
            tail = f.readline().strip()[4:]

            # TGT:Lord Roem
            head = f.readline().strip()[4:]

            # VOT:1
            vote = int(f.readline().strip()[4:])

            # RES:1
            res = int(f.readline().strip()[4:])

            # YEA:2013
            year = int(f.readline().strip()[4:])

            #DAT:19:53, 25 January 2013
            date = f.readline().strip()[4:]

            # TXT:'''Support''' per [[WP:DEAL]]: clueful, and unlikely to break Wikipedia.
            txt = f.readline().strip()[4:]

            # kill blank line
            f.readline()

            # Process the entry
            # index users
            if tail not in usermap:
                usermap[tail] = u
                u += 1
            if head not in usermap:
                usermap[head] = u
                u += 1

            if vote != 0:
                # add vote edge
                A[0, usermap[tail], usermap[head]] = vote

                # add to incidence matrix

                B[v, usermap[tail]] = 1
                B[v, usermap[head]] = 1

                # vectorize text using the hashing trick
                #try:
                #    commentcount[(tail,head)] += 1
                #except KeyError:
                #    commentcount[(tail,head)] = 1
                #features = np.zeros(n_features)
                #for token in txt.strip().split(' '):
                #    features[hash(token) % n_features] += 1
                #A[1:, usermap[tail], usermap[head]] = features

                A[1:, usermap[tail], usermap[head]] = np.asarray(vectorizer.transform([txt]).todense())[0]

                X[usermap[head],0] = resmap[res]

                # treat the result as the node class
                Y[v, votemap[vote]] = 1
                v += 1

    # normalize features
    #for edge in commentcount:
    #    A[1:, edge[0], edge[1]] /= commentcount[edge]

    print u
    print v

    assert len(usermap) == n_nodes
    assert u == n_nodes
    assert v == n_edges

    return A, B, X, Y






