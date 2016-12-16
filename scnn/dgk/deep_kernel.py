"""
Author: Pinar Yanardag (ypinar@purdue.edu)
Please refer to: http://web.ics.purdue.edu/~ypinar/kdd for more details.
"""
import numpy as np
import multiprocessing as mp
import networkx as nx
from gensim.models import Word2Vec
from itertools import chain, combinations
from collections import defaultdict
import sys, copy, time, math, pickle
import itertools
import scipy.io
import pynauty
import random
from scipy.spatial.distance import pdist, squareform

#random.seed(314124)
#np.random.seed(2312312)

def load_data(ds_name):
    f = open(data_dir + "/%s.graph"%ds_name, "r")
    data = pickle.load(f)
    graph_data = data["graph"]
    labels = data["labels"]
    labels  = np.array(labels, dtype = np.float)
    scipy.io.savemat(output_dir + "/%s_labels.mat"%ds_name, mdict={'label': labels})
    return graph_data

def load_graph(nodes):
    max_size = []
    for i in nodes.keys():
        max_size.append(i)
    for nidx in nodes:
        for neighbor in nodes[nidx]["neighbors"]:
            max_size.append(neighbor)
    size = max(max_size)+1
    am = np.zeros((size, size))
    for nidx in nodes:
        for neighbor in nodes[nidx]["neighbors"]:
            am[nidx][neighbor] = 1
    return am

def build_graphlet_corpus(ds_name, min_k, max_k, num_graphlets, samplesize):
    # if no graphlet is found in a graph, we will fall back to 0th graphlet of size k
    fallback_map = {1: 1, 2: 2, 3: 4, 4: 8, 5: 19, 6: 53, 7: 209, 8: 1253, 9: 13599}
    data = load_data(ds_name)
    len_data = len(data)
    canonical_map, weight_map = get_maps(num_graphlets)
    stat_arr = []
    vocabulary = set()
    corpus = []
    # randomly sample graphlets
    graph_map = {}
    graph_map2 = {}
    for gidx in range(len(data)):
        am = load_graph(data[gidx])
        m = len(am)
        count_map = {}
        base_map = {}
        tmp_corpus = []
        if m >=num_graphlets:
            for j in range(samplesize):
                r =  np.random.permutation(range(m))
                for n in [num_graphlets]: #range(1,num_graphlets+1):
                    window = am[np.ix_(r[0:n],r[0:n])]
                    g_type = canonical_map[get_graphlet(window, n)]
                    graphlet_idx = g_type["idx"]
                    level = g_type["n"]
                    count_map[graphlet_idx] = count_map.get(graphlet_idx, 0) + 1
                    # for each node, pick a node it is connected to, and place a non-overlapping? window
                    tmp_corpus.append(graphlet_idx)
                    vocabulary.add(graphlet_idx)
                    for node in r[0:n]:
                        # place a window to for each node in the original window
                        new_n_arr = r[n:][0:n-1] # select n-1 nodes
                        r2 = np.array(list(new_n_arr) + [node])
                        window2 = am[np.ix_(r2,r2)]
                        g_type2 = canonical_map[get_graphlet(window2, n)]
                        graphlet_idx2 = g_type2["idx"]
                        count_map[graphlet_idx2] = count_map.get(graphlet_idx2, 0) + 1
                        vocabulary.add(graphlet_idx2)
                        tmp_corpus.append(graphlet_idx2)
            corpus.append(tmp_corpus)
        else:
            count_map[fallback_map[num_graphlets]] = samplesize # fallback to 0th node at that level
        graph_map[gidx] = count_map
        print "Graph: %s #nodes: %s  total samples: %s"%(gidx, len(data[gidx]), sum(graph_map[gidx].values()))

    print "Total size of the corpus: ", len(corpus)
    prob_map = {gidx: {graphlet: count/float(sum(graphlets.values())) \
        for graphlet, count in graphlets.iteritems()} for gidx, graphlets in graph_map.iteritems()}
    num_graphs = len(prob_map)
    return corpus, vocabulary, prob_map, num_graphs

def get_graphlet(window, nsize):
    """
    This function takes the upper triangle of a nxn matrix and computes its canonical map
    """
    adj_mat = {idx: [i for i in list(np.where(edge)[0]) if i!=idx] for idx, edge in enumerate(window)}

    g = pynauty.Graph(number_of_vertices=nsize, directed=False, adjacency_dict = adj_mat)
    cert = pynauty.certificate(g)
    return cert

def get_maps(n):
    # canonical_map -> {canonical string id: {"graph", "idx", "n"}}
    file_counter = open(data_dir2 + "canonical_maps/canonical_map_n%s.p"%n, "rb")
    canonical_map = pickle.load(file_counter)
    file_counter.close()
    # weight map -> {parent id: {child1: weight1, ...}}
    file_counter = open(data_dir2 + "graphlet_counter_maps/graphlet_counter_nodebased_n%s.p"%n, "rb")
    weight_map = pickle.load(file_counter)
    file_counter.close()
    weight_map = {parent: {child: weight/float(sum(children.values())) for child, weight in children.iteritems()} for parent, children in weight_map.iteritems()}
    child_map = {}
    for parent, children in weight_map.iteritems():
        for k,v in children.iteritems():
            if k not in child_map:
                child_map[k] = {}
            child_map[k][parent] = v
    weight_map = child_map
    return canonical_map, weight_map

def adj_wrapper(g):
    am_ = g["al"]
    size =  max(np.shape(am_))
    am = np.zeros((size, size))
    for idx, i in enumerate(am_):
        for j in i:
            am[idx][j-1] = 1
    return am

def build_sp_corpus(ds_name):
    graph_data = load_data(ds_name)
    vocabulary = set()
    prob_map = {}
    graphs = {}
    corpus = []
    # compute MLE seperately for time-measuring purposes
    for gidx, graph in graph_data.iteritems():
        prob_map[gidx] = {}
        graphs[gidx] = []
        label_map = [graph[nidx]["label"] for nidx in sorted(graph.keys())]
        # construct networkX graph
        G=nx.Graph()
        for nidx in graph:
            edges = graph[nidx]["neighbors"]
            if len(edges) == 0: # some nodes don't have any neighbors
                continue
            for edge in edges:
                G.add_edge(nidx, edge)
        # get all pairs shortest paths
        all_shortest_paths = nx.all_pairs_shortest_path(G) # nx.floyd_warshall(G)
        # traverse all paths and subpaths
        tmp_corpus = []
        for source, sink_map in all_shortest_paths.iteritems():
            for sink, path in sink_map.iteritems():
                sp_length = len(path)-1
                label = "_".join(map(str, sorted([label_map[source][0],label_map[sink][0]]))) + "_" + str(sp_length)
                tmp_corpus.append(label)
                prob_map[gidx][label] = prob_map[gidx].get(label, 0) + 1
                graphs[gidx].append(label)
                vocabulary.add(label)
        corpus.append(tmp_corpus)
    # note that in MLE version, prob_map is not normalized.
    prob_map = {gidx: {path: count/float(sum(paths.values())) \
        for path, count in paths.iteritems()} for gidx, paths in prob_map.iteritems()}
    num_graphs = len(prob_map)
    return corpus, vocabulary, prob_map, num_graphs

def get_label(s):
    #return "_".join([str(i[0]) for i in s])
    x = sorted([s[0], s[-1]])
    x = str(x[0][0]) + "_" + str(x[-1][0]) + "_" + str(len(s))
    return x

def build_wl_corpus(ds_name, max_h):
    graph_data = load_data(ds_name)
    labels = {}
    label_lookup = {}
    label_counter = 0
    vocabulary = set()
    num_graphs = len(graph_data)
    max_window = []
    wl_graph_map = {it: {gidx: defaultdict(lambda: 0) for gidx in range(num_graphs)} for it in range(-1, max_h)}
    sim_map = {}

    # initial labeling
    for gidx in range(num_graphs):
        labels[gidx] = np.zeros(len(graph_data[gidx]), dtype = np.int32)
        for node in range(len(graph_data[gidx])):
            label = graph_data[gidx][node]["label"]
            if not label_lookup.has_key(label):
                label_lookup[label] = label_counter
                labels[gidx][node] = label_counter
                label_counter += 1
            else:
                labels[gidx][node] = label_lookup[label]
            wl_graph_map[-1][gidx][label_lookup[label]] = wl_graph_map[-1][gidx].get(label_lookup[label], 0) + 1
    compressed_labels = copy.deepcopy(labels)
    # WL iterations started
    for it in range(max_h):
        label_lookup = {}
        label_counter = 0
        for gidx in range(num_graphs):
            for node in range(len(graph_data[gidx])):
                node_label = tuple([labels[gidx][node]])
                neighbors = graph_data[gidx][node]["neighbors"]
                if len(neighbors) > 0:
                    neighbors_label = tuple([labels[gidx][i] for i in neighbors])
                    #node_label =  str(node_label) + "-" + str(sorted(neighbors_label))
                    node_label = tuple(tuple(node_label) + tuple(sorted(neighbors_label)))
                if not label_lookup.has_key(node_label):
                    label_lookup[node_label] = str(label_counter)
                    compressed_labels[gidx][node] = str(label_counter)
                    label_counter += 1
                else:
                    compressed_labels[gidx][node] = label_lookup[node_label]
                wl_graph_map[it][gidx][label_lookup[node_label]] = wl_graph_map[it][gidx].get(label_lookup[node_label], 0) + 1
        print "Number of compressed labels at iteration %s: %s"%(it, len(label_lookup))
        labels = copy.deepcopy(compressed_labels)

    # merge the following code into the loop above
    graphs = {}
    prob_map = {}
    corpus = []
    for it in range(-1, max_h):
        for gidx, label_map in wl_graph_map[it].iteritems():
            if gidx not in graphs:
                graphs[gidx] = []
                prob_map[gidx] = {}
            for label_, count in label_map.iteritems():
                label = str(it) + "+" + str(label_)
                for c in range(count):
                    graphs[gidx].append(label)
                vocabulary.add(label)
                prob_map[gidx][label] = count

    corpus = [graph for gidx, graph in graphs.iteritems()]
    vocabulary = sorted(vocabulary)
    return corpus, vocabulary, prob_map,  num_graphs

def l2_norm(vec):
    #return np.sqrt(sum(i*i for i in vec))
    return  np.sqrt(np.dot(vec, vec))

def main(num_dimensions=1, # any integer > 0
         kernel_type=1, # 1 (deep, l2) or 2 (deep, M), 3 (MLE)
         feature_type=3, # 1 (graphlet), 2 (SP), 3 (WL)
         ds_name='nci1', # dataset name
         window_size=1, # any integer > 0
         ngram_type=1, # 1 (skip-gram), 0 (cbow)
         sampling_type=1, # 1 (hierarchical sampling), 0 (negative sampling)
         graphlet_size=1, # any integer > 0
         sample_size=1 # any integer > 0
         ):
    global data_dir
    global output_dir
    LOCAL_RUN = True

    # hyperparameters
    run_parallel = 1 # 1 activates word2vec's parallel option

    # graph kernel parameters
    min_k = 7 # for graphlet
    max_k = 7 # for graphlet
    max_h = 2 # for WL

    print "Dataset: %s\n\nWord2vec Parameters:\nDimension: %s\nWindow size: %s\nNgram type: %s\nSampling type: %s\nParallel: %s\
            \n\nKernel-related Parameters:\nKernel type: %s\nFeature type: %s\nWL height: %s\nGraphlet size: %s\nSample size: %s\n"\
            %(ds_name, num_dimensions, window_size, ngram_type, sampling_type,run_parallel, kernel_type, feature_type, max_h, graphlet_size, sample_size)

    if LOCAL_RUN:
        output_dir = "../results"
        data_dir = "data/"
    else:
        output_dir = "/scratch/lustreA/y/ypinar"
        data_dir = "/scratch/lustreA/y/ypinar/datasets/"
        data_dir2 = "/home/ypinar/graphlet-mkl/Data/"

    # STEP 1: Build corpus
    start = time.time()
    if feature_type == 1:
        # terms are graphlets
        corpus, vocabulary, prob_map, num_graphs = build_graphlet_corpus(ds_name, min_k, max_k, graphlet_size, sample_size)
    elif feature_type == 2:
        # terms are labeled shortest paths
        corpus, vocabulary, prob_map, num_graphs = build_sp_corpus(ds_name)
    elif feature_type == 3:
        # terms are labeled sub-trees
        corpus, vocabulary, prob_map,  num_graphs = build_wl_corpus(ds_name, max_h)
    else:
        raise Exception("Unknown feature type!")
    end = time.time()
    vocabulary = list(sorted(vocabulary))
    print "Corpus construction total time: %g vocabulary size: %s"%(end-start, len(vocabulary))

    # STEP 2: learn hidden representations
    # get word2vec representations
    start = time.time()
    model = Word2Vec(corpus, size=num_dimensions, window=window_size, min_count=0, sg=ngram_type, hs=sampling_type)
    end = time.time()
    print "M matrix total time: %g"%(end-start)

    # STEP 3: compute the kernel
    K = np.zeros((num_graphs, num_graphs))
    if kernel_type == 1:
        # deep w/l2 norm
        P = np.zeros((num_graphs, len(vocabulary)))
        for i in range(num_graphs):
            for jdx, j in enumerate(vocabulary):
                P[i][jdx] = prob_map[i].get(j,0)
        M = np.zeros((len(vocabulary), len(vocabulary)))
        for idx,i in enumerate(vocabulary):
            M[idx][idx] = l2_norm(model[i])
        K = (P.dot(M)).dot(P.T)
    elif kernel_type == 2:
        P = np.zeros((num_graphs, len(vocabulary)))
        for i in range(num_graphs):
            for jdx, j in enumerate(vocabulary):
                P[i][jdx] = prob_map[i].get(j,0)
        M = np.zeros((len(vocabulary), len(vocabulary)))
        for i in range(len(vocabulary)):
            for j in range(len(vocabulary)):
                M[i][j] = np.dot(model[vocabulary[i]], model[vocabulary[j]])
        K = (P.dot(M)).dot(P.T)
    elif kernel_type == 3:
        #MLE
        P = np.zeros((num_graphs, len(vocabulary)))
        for i in range(num_graphs):
            for jdx, j in enumerate(vocabulary):
                P[i][jdx] = prob_map[i].get(j,0)
        K = P.dot(P.T)

    return K
    # the following computed kernel can be directly fed to libsvm library
    #scipy.io.savemat("%s/deep_kernel_%s_k%s_d%s_f%s_w%s_ngram%s_sampling%s_gsize%s_samplesize%s.mat"%(output_dir, ds_name, kernel_type, num_dimensions, feature_type, window_size, ngram_type, sampling_type, graphlet_size, sample_size), mdict={'kernel': K})

