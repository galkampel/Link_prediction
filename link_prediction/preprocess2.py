
import networkx as nx
import numpy as np
import random
from time import time
from file2obj import save_to_gzip
from joblib import Parallel,delayed
from itertools import chain
from multiprocessing import cpu_count,Manager


def neg_between_edges(edges,nodes1,nodes2):
    N = len(edges)
    neg_edges = []
    i = 0
    while i < N:
        node1 = np.random.choice(nodes1)
        node2 = np.random.choice(nodes2)
        edge = node1,node2
        if edge not in edges:
            i += 1
            neg_edges.append(edge)
    return neg_edges

class Preprocess(object):
    def __init__(self,G,community_dict,between_edges_dict,frac_within,frac_between, seed=1):
        self.frac_within = frac_within #  #test size positive edges and #test size neg edges
        self.frac_between = frac_between
        #fields from the Graph class
        self.G = G
        self.num_cores = cpu_count()

        # manager = Manager()
        self.train_neg_edges = {}
        self.train_neg_edges['within'] = {}
        self.train_neg_edges['between'] = {}
        # self.train_neg_edges = manager.dict()
        # self.train_neg_edges['within'] = manager.dict()
        # self.train_neg_edges['between'] = manager.dict()

        self.community_dict = community_dict
        self.between_edges_dict = between_edges_dict
#num_chunks: number of chunks to create positive test set
        # self.num_chunks = num_chunks
### split negative labels to randomize and in-cluster
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)



    def parallel_within_neg_edges(self,communities_dict_chunk,test_neg_edges,test_pos_edges):
        test_neg_within = {}
        for label,nodes in communities_dict_chunk.items():
            G_sub = self.G.subgraph(nodes)
            num_edges = int(self.frac_within * len(G_sub.edges()))
            test_neg_within.setdefault(label,[]).extend(self.create_test_neg_within_edges(nodes,num_edges))
        return test_neg_within
            #get positive within
        # test_neg_edges['within'].update(test_neg_within)
        # test_pos_edges['within'].update(test_pos_within)

    def parallel_within_pos_edges(self,communities_dict_chunk,test_neg_edges,test_pos_edges):
        test_pos_within = {}
        for label,nodes in communities_dict_chunk.items():
            G_sub = self.G.subgraph(nodes)
            num_edges = int(self.frac_within * len(G_sub.edges()))
            edges = self.create_positive_edge_within(G_sub,num_edges)
            test_pos_within.setdefault(label,[]).extend(edges)
        return test_pos_edges


    def parallel_between_neg_edges(self,communities_chunk,community_dict,test_neg_edges,test_pos_edges):
        tmp_dict_neg = {}
        frac_between = self.frac_between
        for communities,edges in communities_chunk.items():
            com1,com2 = communities
            num_edges = int(frac_between * len(edges))
            nodes1,nodes2 = community_dict[com1],community_dict[com2]
            tmp_dict_neg[communities] = self.create_neg_between_edges(nodes1,nodes2,num_edges)
        return tmp_dict_neg

    def parallel_between_pos_edges(self,communities_chunk,community_dict,test_neg_edges,test_pos_edges):
        tmp_dict_pos = {}
        frac_between = self.frac_between
        for communities,edges in communities_chunk.items():
            num_edges = int(frac_between * len(edges))
            tmp_dict_pos[communities] = self.create_positive_edge_between(edges,num_edges)
        return tmp_dict_pos

    def slice_dict(self,d,num_chunks):
        items = list(d.items())
        sliced_dict = list(map(dict, list(items[i::num_chunks] for i in range(num_chunks))))
        return sliced_dict
## filename- the file to save the raw test set
    def set_new_G(self,filename):
        # test_neg_edges = {'within':{},'between':{}}
        # test_pos_edges ={'within':{},'between':{}}
        G = self.G
        community_dict = self.community_dict
        # manager = Manager()
        # test_neg_edges = manager.dict()
        # test_neg_edges['within'] = manager.dict()
        # test_neg_edges['between'] = manager.dict()
        # test_pos_edges = manager.dict()
        # test_pos_edges['within'] = manager.dict()
        # test_pos_edges['between'] = manager.dict()
        # start = time()
        test_neg_edges = {}
        test_neg_edges['within'] = {}
        test_neg_edges['between'] = {}
        test_pos_edges = {}
        test_pos_edges['within'] = {}
        test_pos_edges['between'] = {}
        num_cores = min(self.num_cores,len(community_dict.keys()))
        sliced_community_dict = self.slice_dict(community_dict,num_cores)
        pos_edges_dicts = Parallel(n_jobs=self.num_cores)(delayed(self.parallel_within_pos_edges)(communities_dict_chunk,test_neg_edges,test_pos_edges)
                                   for communities_dict_chunk in sliced_community_dict)
        for el in pos_edges_dicts:
            test_pos_edges['within'].update(el)

        neg_edges_dicts = Parallel(n_jobs=self.num_cores)(delayed(self.parallel_within_neg_edges)(communities_dict_chunk,test_neg_edges,test_pos_edges)
                                   for communities_dict_chunk in sliced_community_dict)
        for el in neg_edges_dicts:
            test_neg_edges['within'].update(el)

        # print('It took {:.3f} seconds to create neg within edges'.format(time()-start))
        # test_pos_edges['within'] = dict(test_pos_edges['within'])
        # test_neg_edges['within'] = dict(test_neg_edges['within'])

        # print('test pos within non-unique equlas to test pos within unique?\n{}'.format(len(list(chain(*test_pos_edges['within'].values()))) == len(set(list(chain(*test_pos_edges['within'].values()))))))
        G.remove_edges_from(list(chain(*test_pos_edges['within'].values())))
        between_edges_dict = self.between_edges_dict
        # start = time()
        num_cores = min(self.num_cores,len(between_edges_dict.keys()))
        sliced_between_edges_dict = self.slice_dict(between_edges_dict,num_cores)
        pos_edges_dicts = Parallel(n_jobs=self.num_cores)(delayed(self.parallel_between_pos_edges)(communities_chunk,community_dict,test_neg_edges,test_pos_edges)
                                    for communities_chunk in sliced_between_edges_dict)
        for el in pos_edges_dicts:
            test_pos_edges['between'].update(el)

        neg_edges_dicts = Parallel(n_jobs=self.num_cores)(delayed(self.parallel_between_neg_edges)(communities_chunk,community_dict,test_neg_edges,test_pos_edges)
                                    for communities_chunk in sliced_between_edges_dict)
        for el in neg_edges_dicts:
            test_neg_edges['between'].update(el)

        # print('test pos between non-unique equals to test pos between unique?\n{}'.format(len(list(chain(*test_pos_edges['between'].values()))) == len(set(list(chain(*test_pos_edges['between'].values()))))))
        G.remove_edges_from(list(chain(*test_pos_edges['between'].values())))
        # frac_between = self.frac_between
        # for communities,edges in between_edges_dict.items():
        #     com1,com2 = communities
        #     num_edges = int(frac_between * len(edges))
        #     nodes1,nodes2 = community_dict[com1],community_dict[com2]
        #     test_neg_edges['between'].setdefault(communities,[]).extend(self.create_test_neg_between_edges(nodes1,nodes2,num_edges))
            # chunk_size = int(num_edges / num_chunks)
            # last_chunk = num_edges - (num_chunks-1)*chunk_size
            # chunks = [chunk_size for i in range(num_chunks-1)]
            # chunks.append(last_chunk)
            # for chunk in chunks:
            #     pos_edges =
            # test_pos_edges['between'].setdefault(communities,[]).extend(self.create_positive_edge_between(edges,num_edges))


## saving test to file (before exerting node2vec) under dict
        # print('It took {:.3f} seconds to create neg community edges'.format(time()-start))
        test_pos_edges = dict(test_pos_edges)
        test_neg_edges = dict(test_neg_edges)
        obj = {'pos': test_pos_edges,'neg':test_neg_edges}
        save_to_gzip(obj,'{}.gz'.format(filename))


    def create_positive_edge_within(self, G_sub ,size):
        edges = list(G_sub.edges())
        mst_edges = set(nx.minimum_spanning_edges(G_sub,data=False))
        cand_edges = list(set(edges).difference(mst_edges))
        if len(cand_edges) < size:
            size = len(cand_edges)
        range_edges = range(len(cand_edges))
        G_copy = G_sub.copy()
        # G = self.G
        # while True:
        idxes =  np.random.choice(range_edges,size,replace=False)
        rel_edges = [cand_edges[idx] for idx in idxes]
        G_copy.remove_edges_from(rel_edges)
        if nx.number_connected_components(G_copy) == 1:
            # print('num edges before removal = {}'.format(len(G.edges())))
            # G.remove_edges_from(cand_edges)
            # print('num edges after removal = {}'.format(len(G.edges())))
            return rel_edges
        else:
            print('error!!!!')
            exit()
            # G_copy = G_sub.copy()


## always remains connected
    def create_positive_edge_between(self,edges,num_edges):
        range_edges = range(len(edges))
        idxes = np.random.choice(range_edges,num_edges,replace=False)
        removed_edges = [edges[idx] for idx in idxes]
        # G = self.G
        # print('num edges before removal = {}'.format(len(G.edges())))
        # G.remove_edges_from(removed_edges)
        # print('num edges after removal = {}'.format(len(G.edges())))
        return removed_edges


    def create_test_neg_within_edges(self,nodes,num_neg_edges):
        G = self.G
        neg_edges =[]
        count = 0
        while count < num_neg_edges:
            candidtate = tuple(np.random.choice(nodes,2,replace=False))
            if not G.has_edge(*candidtate) and candidtate not in neg_edges:
                neg_edges.append(candidtate)
                count += 1
        # print('It took {:.3f} seconds to create test negative within edges'.format(time()-start))
        return neg_edges


    def create_neg_between_edges(self,nodes1,nodes2,num_neg_edges):
        G = self.G
        # G_sub_edges = set(G.subgraph(nodes1+nodes2).edges())
        # all_edges = set(nx.complete_graph(nodes1+nodes2).edges())
        # rel_edges = list(all_edges.difference(G_sub_edges))
        # range_rel_edges = range(len(rel_edges))
        # idxes = np.random.choice(range_rel_edges,num_neg_edges,replace=False)
        # neg_edges = [rel_edges[idx] for idx in idxes]
        neg_edges = []
        count = 0
        while count < num_neg_edges:
            candidtate = np.random.choice(nodes1),np.random.choice(nodes2)
            if not G.has_edge(*candidtate) and candidtate not in neg_edges:
                neg_edges.append(candidtate)
                count += 1
        return neg_edges


    def parallel_train_neg_between(self,between_edges_dict_chunk,community_dict):
        d = {}
        for communities,edges in between_edges_dict_chunk.items():
            nodes1,nodes2 = community_dict[communities[0]], community_dict[communities[1]]
            d[communities] = self.create_neg_between_edges(nodes1,nodes2,len(edges))#neg_between_edges(edges,nodes1,nodes2)

        return d
        # self.train_neg_edges['between'].update(d)
        # self.train_neg_edges['between'].setdefault(communities,[]).extend(neg_between_edges(edges,nodes1,nodes2))




    def create_train_neg_between_edges(self):
        community_dict = self.community_dict
        num_cores = min(self.num_cores,len(self.between_edges_dict.keys()))
        sliced_between_edges_dict = self.slice_dict(self.between_edges_dict,num_cores)
        neg_between_dicts = Parallel(n_jobs=self.num_cores)(delayed(self.parallel_train_neg_between)(between_edges_dict_chunk,community_dict)
                                        for between_edges_dict_chunk in sliced_between_edges_dict)
        for el in neg_between_dicts:
            self.train_neg_edges['between'].update(el)
        # between_edges_dict = self.between_edges_dict
        # for communities,edges in between_edges_dict.items():
        #     nodes1,nodes2 = community_dict[communities[0]], community_dict[communities[1]]
        #     self.train_neg_edges['between'].setdefault(communities,[]).extend(neg_between_edges(edges,nodes1,nodes2))
        # train_neg_edges = self.train_neg_edges
        # for communities,edges in between_edges_dict.items():
        #     nodes1,nodes2 = community_dict[communities[0]], community_dict[communities[1]]
        #     train_neg_edges['between'].setdefault(communities,[]).extend(neg_between_edges(edges,nodes1,nodes2))

    def parallel_train_neg_within(self,community_dict_chunk):
        G = self.G
        d = {}
        for label,nodes in community_dict_chunk.items():
            N = len(list(G.subgraph(nodes).edges()))
            d[label] = self.create_test_neg_within_edges(nodes,N)
        return d

    def create_train_neg_within_edges(self):
        # train_neg_edges = self.train_neg_edges
        # community_dict = self.community_dict
        # G = self.G
        # for label,nodes in community_dict.items():
        #     N = len(list(G.subgraph(nodes).edges()))
        #     train_neg_edges['within'].setdefault(label,[]).extend(self.create_test_neg_within_edges(nodes,N))
        num_cores = min(self.num_cores,len(self.community_dict.keys()))
        sliced_community_dict = self.slice_dict(self.community_dict,num_cores)
        neg_between_dicts = Parallel(n_jobs=self.num_cores)(delayed(self.parallel_train_neg_within)(community_dict_chunk)
                                        for community_dict_chunk in sliced_community_dict)
        for el in neg_between_dicts:
            self.train_neg_edges['within'].update(el)

    def parallel_within_community_edges(self,community_dict_chunk):

        tmp_within_community_edges = {}
        G = self.G
        for key,nodes in community_dict_chunk.items():
            tmp_within_community_edges[key] = list(G.subgraph(nodes).edges())


    def parallel_train_pos_within(self,community_dict_chunk,within_community_edges):

        tmp_within_edges = {}
        for key,nodes in community_dict_chunk.items():
            tmp_within_edges[key] = list(self.G.subgraph(nodes).edges())
        within_community_edges.update(tmp_within_edges)

    def get_within_community_edges(self):
        community_dict= self.community_dict
        G = self.G
        within_community_edges = {}
        for key,nodes in community_dict.items():
            within_community_edges[key] = list(G.subgraph(nodes).edges())
        # sliced_community_dict = self.slice_dict(community_dict)
        # manager = Manager()
        # within_community_edges = manager.dict()
        #
        # start = time()
        # Parallel(n_jobs=self.num_cores)(delayed(self.parallel_train_pos_within)(community_dict_chunk,within_community_edges)
        #                                 for community_dict_chunk in sliced_community_dict)
        # print('It took {:.3f} seconds to parallelize within community dict'.format(time()-start))
        # within_community_edges = dict(within_community_edges)
        return within_community_edges

    def get_between_edges_dict(self):
        return self.between_edges_dict

    def get_G(self):
        return self.G

    def save_train_obj(self,filename):
        start = time()
        self.create_train_neg_between_edges()
        self.train_neg_edges['between'] = dict(self.train_neg_edges['between'])
        print('it took {:.3f} seconds to create train neg betweeen edges'.format(time()-start))
        start = time()
        self.create_train_neg_within_edges()
        self.train_neg_edges['within'] = dict(self.train_neg_edges['within'])

        self.train_neg_edges = dict(self.train_neg_edges)
        print('it took {:.3f} seconds to create train neg within edges'.format(time()-start))
        start = time()
        train_pos_edges = {'within': self.get_within_community_edges(), 'between':self.get_between_edges_dict()}
        obj = {'pos': train_pos_edges,'neg': self.train_neg_edges}
        print('it took {:.3f} seconds to get positive edges '.format(time()-start))
        # print("num positive between link: {}\nnum negative between links: {}".format(
        #     np.sum(len(nodes) for nodes in (train_pos_edges['between'].values())),
        #       np.sum(len(nodes) for nodes in self.train_neg_edges['between'].values())))
        # print("num positive within link: {}\nnum negative within links: {}".format(
        #     np.sum(len(nodes) for nodes in (train_pos_edges['within'].values())),
        #       np.sum(len(nodes) for nodes in self.train_neg_edges['within'].values())))
        save_to_gzip(obj,'{}.gz'.format(filename))



    def save_new_G(self,filename):
        nx.write_edgelist(self.G,filename+'{}.gz'.format(self.seed),data=False)
