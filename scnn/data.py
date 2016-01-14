__author__ = 'jatwood'

import numpy as np

import theano
import theano.tensor as T

import sys, os, glob

import networkx as nx
import re
import pickle

import hashlib


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

    return (adj.astype('float32'), features.astype('float32'), labels.astype('int32'))
