"""Graph utilities."""

# from time import time
import networkx as nx
# import pickle as pkl
import numpy as np
import scipy.sparse as sp
from file2obj import read_gzip_object
__author__ = "Zhang Zhengyan"
__email__ = "zhangzhengyan14@mails.tsinghua.edu.cn"


class Graph(object):
    def __init__(self,G= None,num_neg_classes=1):
        self.G = G
        self.look_up_dict = {}
        self.look_back_list = []
        self.node_size = 0
### addition - dictionary- for each label, store the relevant nodes
        self.label_dict = {}
### num_classes- if given -> number of classes is 2*#C instead of #C + 1
        self.num_neg_classes = num_neg_classes

    def encode_node(self):
        look_up = self.look_up_dict
        look_back = self.look_back_list
        for node in self.G.nodes():
            look_up[node] = self.node_size
            look_back.append(node)
            self.node_size += 1
            self.G.nodes[node]['status'] = ''




    def read_adjlist(self, filename):
        """ Read graph from adjacency file in which the edge must be unweighted
            the format of each line: v1 n1 n2 n3 ... nk
            :param filename: the filename of input file
        """
        self.G = nx.read_adjlist(filename, create_using=nx.DiGraph())
        for i, j in self.G.edges():
            self.G[i][j]['weight'] = 1.0
        self.encode_node()

    def read_edgelist(self, filename, weighted=False, directed=False,to_weighted=False):
        self.G = nx.DiGraph()
        np.random.seed(1)
        if directed:
            ### manual weights
            if to_weighted:
                def read_unweighted(l):
                    src, dst = l.split()
                    self.G.add_edge(src, dst)
                    self.G[src][dst]['weight'] = 9 * np.random.random() + 1
            else:
                def read_unweighted(l):
                    src, dst = l.split()
                    self.G.add_edge(src, dst)
                    self.G[src][dst]['weight'] = 1.0

            def read_weighted(l):
                src, dst, w = l.split()
                self.G.add_edge(src, dst)
                self.G[src][dst]['weight'] = float(w)
        else:

            if to_weighted:
                def read_unweighted(l):
                    src, dst = l.split()
                    self.G.add_edge(src, dst)
                    self.G.add_edge(dst, src)
                    val = 8 * np.random.random() + 1
                    self.G[src][dst]['weight'] = val
                    val2 = val + np.random.standard_normal()
                    if val2 < 1.0:
                        val2 = 1.0
                    self.G[dst][src]['weight'] = val2
            else:
                def read_unweighted(l):
                    src, dst = l.split()
                    self.G.add_edge(src, dst)
                    self.G.add_edge(dst, src)
                    self.G[src][dst]['weight'] = 1.0
                    self.G[dst][src]['weight'] = 1.0

            def read_weighted(l):
                src, dst, w = l.split()
                self.G.add_edge(src, dst)
                self.G.add_edge(dst, src)
                self.G[src][dst]['weight'] = float(w)
                self.G[dst][src]['weight'] = float(w)

        func = read_unweighted
        if weighted:
            func = read_weighted
        edges = read_gzip_object(filename).edges()
        for u,v in edges:
            func('{} {}'.format(u,v))
        #change from 'r' to 'rb'
        # fin = open(filename, 'r')
        # func = read_unweighted
        # if weighted:
        #     func = read_weighted
        # while 1:
        #     l = fin.readline()
        #     print(l)
        #     if l == '':
        #         break
        #     func(l)
        # fin.close()
        # self.encode_node()

    def read_node_label(self, filename):
        fin = open(filename, 'r')
        label_dict = self.label_dict
        num_neg_classes = self.num_neg_classes
        while 1:
            l = fin.readline()
            if l == '':
                break
            vec = l.split()
# save label = 0 to negative edges or #num_classes
            label = int(vec[1:]) + num_neg_classes
            # label = int(vec[1:])
            self.G.nodes[vec[0]]['label'] = label
            label_dict[label] = label_dict.setdefault(label, []).append(self.G.nodes[vec[0]])
        fin.close()

    def read_node_features(self, filename):
        fin = open(filename, 'r')
        for l in fin.readlines():
            vec = l.split()
            self.G.nodes[vec[0]]['feature'] = np.array([float(x) for x in vec[1:]])
        fin.close()

    def read_node_status(self, filename):
        fin = open(filename, 'r')
        while 1:
            l = fin.readline()
            if l == '':
                break
            vec = l.split()
            self.G.nodes[vec[0]]['status'] = vec[1] # train test valid
        fin.close()

    def read_edge_label(self, filename):
        fin = open(filename, 'r')
        while 1:
            l = fin.readline()
            if l == '':
                break
            vec = l.split()
            self.G[vec[0]][vec[1]]['label'] = vec[2:]
        fin.close()

