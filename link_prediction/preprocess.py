
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
    def __init__(self,G,community_dict,between_edges_dict,multiplier,frac_within,frac_between, seed=1):
        self.frac_within = frac_within #  #test size positive edges and #test size neg edges
        self.frac_between = frac_between
        #fields from the Graph class
        self.G = G
        self.num_cores = cpu_count()
        self.multiplier = multiplier
        manager = Manager()
        self.train_neg_edges = manager.dict()
        self.train_neg_edges['within'] = manager.dict()
        self.train_neg_edges['between'] = manager.dict()

        self.community_dict = community_dict
        self.between_edges_dict = between_edges_dict
#num_chunks: number of chunks to create positive test set
        # self.num_chunks = num_chunks
### split negative labels to randomize and in-cluster
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)



    def parallel_within_edges(self,communities_dict_chunk,test_neg_edges,
                              test_pos_edges,thres = 8):
        test_neg_within = {}
        test_pos_within = {}
        for label,nodes in communities_dict_chunk.items():

            G_sub = self.G.subgraph(nodes)
            num_edges = len(G_sub.edges())
            if num_edges < thres:
                continue

            num_edges = int(self.frac_within * len(G_sub.edges()))
            num_pos_edges = 0
            for subgraph in nx.connected_component_subgraphs(G_sub):
                if len(subgraph.edges()) < thres:
                    continue
                edges = self.create_positive_edge_within(subgraph, num_pos_edges, num_edges)
                if len(edges) == 0:
                    continue
                num_pos_edges += len(edges)
                test_pos_within.setdefault(label, []).extend(edges)


            num_edges = num_pos_edges
            neg_edges = self.create_test_neg_within_edges(nodes, num_edges)
            test_neg_within.setdefault(label, []).extend(neg_edges)


            #get positive within
            # print('G sub has {} nodes and {} edges'.format(len(G_sub.nodes()),len(G_sub.edges())))

        test_neg_edges['within'].update(test_neg_within)
        test_pos_edges['within'].update(test_pos_within)

    def parallel_between_edges(self,communities_chunk,community_dict,test_neg_edges,test_pos_edges):
        tmp_dict_neg = {}
        tmp_dict_pos = {}
        frac_between = self.frac_between
        for communities,edges in communities_chunk.items():
            com1,com2 = communities
            num_edges = int(frac_between * len(edges))
            nodes1,nodes2 = community_dict[com1],community_dict[com2]
            tmp_dict_pos[communities] = self.create_positive_edge_between(edges,num_edges)
            tmp_dict_neg[communities] = self.create_neg_between_edges(nodes1,nodes2,num_edges)


        test_neg_edges['between'].update(tmp_dict_neg)
        test_pos_edges['between'].update(tmp_dict_pos)


    def slice_dict(self,d,num_chunks):
        items = list(d.items())
        sliced_dict = list(map(dict, list(items[i::num_chunks] for i in range(num_chunks))))
        return sliced_dict

    def update_between_edges_dict(self,test_pos_between):
        between_edges_dict = self.between_edges_dict
        tmp_dict = {}
        for key,els in between_edges_dict.items():
            tmp_els = set(els).difference(set(test_pos_between[key]))
            tmp_dict[key] = list(tmp_els)
        self.between_edges_dict = tmp_dict
## filename- the file to save the raw test set
    def get_outlier(self):
        import pandas as pd
        pd_list = []
        header = ["C", "links in Community C"]
        G = self.G
        for c, v in self.community_dict.items():
            num_edges = len(G.subgraph(v).edges())
            pd_list.append([c, num_edges])
        df = pd.DataFrame.from_records(pd_list, columns=header)
        df.to_csv('C_edges.csv')
        thres = df['links in Community C'].mean() + 2 * df['links in Community C'].std()
        return df[df['links in Community C'] >= thres]["C"].tolist()

    def set_new_G(self,filename):

        G = self.G
        community_dict = self.community_dict
        manager = Manager()
        test_neg_edges = manager.dict()
        test_neg_edges['within'] = manager.dict()
        test_neg_edges['between'] = manager.dict()
        test_pos_edges = manager.dict()
        test_pos_edges['within'] = manager.dict()
        test_pos_edges['between'] = manager.dict()


        Cs_outlier = self.get_outlier()
        tmp_community_dict = {key:nodes for key,nodes in community_dict.items() if key not in Cs_outlier}
        sliced_community_dict = self.slice_dict(tmp_community_dict,len(tmp_community_dict.keys())// 10)
        Parallel(n_jobs=3)(delayed(self.parallel_within_edges)(communities_dict_chunk,test_neg_edges,test_pos_edges)
                                   for communities_dict_chunk in sliced_community_dict)
        for c in Cs_outlier:
            self.parallel_within_edges({c:community_dict[c]},test_neg_edges,test_pos_edges)
        test_pos_edges['within'] = dict(test_pos_edges['within'])
        test_neg_edges['within'] = dict(test_neg_edges['within'])

        print('test_pos_edges[within] has {} communities'.format(len(test_pos_edges['within'])))
        print('G has {} edges'.format(len(G.edges())))
        G.remove_edges_from(list(chain.from_iterable(test_pos_edges['within'].values())))
        print('G has {} edges'.format(len(G.edges())))
        print('Is G connected after within removal?\n{}'.format(nx.is_connected(G)))
        print('G has {} connected components'.format(nx.number_connected_components(G)))

        between_edges_dict = self.between_edges_dict
        mst_edges = set(nx.minimum_spanning_edges(G,data=False))
        complete_edges = set((v,u) for (u,v) in mst_edges )
        self.mst_edges = mst_edges | complete_edges

        num_cores = min(self.num_cores,len(between_edges_dict.keys()))
        sliced_between_edges_dict = self.slice_dict(between_edges_dict,num_cores * self.multiplier)
        Parallel(n_jobs=self.num_cores)(delayed(self.parallel_between_edges)(communities_chunk,community_dict,
                                                                             test_neg_edges,test_pos_edges)
                                    for communities_chunk in sliced_between_edges_dict)

        test_pos_edges['between'] = dict(test_pos_edges['between'])

        # for Cs,edges in test_pos_edges['between'].items() :
        #     for edge in edges:
        #         if edge not in mst_edges:
        #             new_test_between[Cs].append(edge)
        #         else:
        #             num_problematic_edges += 1
        # print('{} edges were removed from positive between'.format(num_problematic_edges))
        # test_pos_edges['between'] = new_test_between

        self.update_between_edges_dict(test_pos_edges['between'])
        test_neg_edges['between'] = dict(test_neg_edges['between'])
        # print('test pos between non-unique equals to test pos between unique?\n{}'.format(len(list(chain(*test_pos_edges['between'].values()))) == len(set(list(chain(*test_pos_edges['between'].values()))))))
        print('Is G connected before between removal?\n{}'.format(nx.is_connected(G)))
        print('G has {} edges'.format(len(G.edges())))

        G.remove_edges_from(list(chain.from_iterable(test_pos_edges['between'].values())))
        print('G has {} edges'.format(len(G.edges())))
        print('Is G connected after between removal?\n{}'.format(nx.is_connected(G)))
        print('G has {} connected components'.format(nx.number_connected_components(G)))

        test_pos_edges = dict(test_pos_edges)
        test_neg_edges = dict(test_neg_edges)
        obj = {'pos': test_pos_edges,'neg':test_neg_edges}
        save_to_gzip(obj,'{}.gz'.format(filename))


    def create_positive_edge_within(self, subgraph,num_pos_edges ,max_size):
        mst_edges = set(nx.minimum_spanning_edges(subgraph,data=False))
        complete_edges = set((v,u) for (u,v) in mst_edges)
        local_mst_edges = mst_edges | complete_edges
        cand_edges = list(set(subgraph.edges()).difference(local_mst_edges))
        size = len(cand_edges)
        if num_pos_edges < max_size and len(cand_edges)+num_pos_edges > max_size:
            size =  len(cand_edges) + num_pos_edges - max_size
        elif num_pos_edges == max_size or size == 0:
            return []
        num_pos_edges += size
        range_edges = range(len(cand_edges))
        # print('range length:\n{}'.format(len(range_edges)))
        G_copy = subgraph.copy()

        # while True:
        idxes =  np.random.choice(range_edges,size,replace=False)
        rel_edges = [cand_edges[idx] for idx in idxes]
        G_copy.remove_edges_from(rel_edges)
        # G = self.G
        if nx.number_connected_components(G_copy) == 1:
            # print('num edges before removal = {}'.format(len(G.edges())))
            # G.remove_edges_from(cand_edges)
            # print('num edges after removal = {}'.format(len(G.edges())))
            return rel_edges
        else:
            print('error!!!!')
            exit()


    def create_positive_edge_between(self,edges,num_edges):
        rel_edges = list(set(edges).difference(self.mst_edges))
        range_edges = range(len(rel_edges))
        size = min(len(rel_edges),num_edges)
        idxes = np.random.choice(range_edges,size,replace=False)
        removed_edges = [rel_edges[idx] for idx in idxes]
        return removed_edges


    def create_test_neg_within_edges(self,nodes,num_neg_edges):
        G = self.G
        G_all = nx.complete_graph(nodes)
        cand_edges = list(set(G_all.edges()).difference(set(G.subgraph(nodes).edges())))

        size = min(num_neg_edges,len(cand_edges))
        # if num_neg_edges >= len(cand_edges):
        #     print('cand edges size = {}\nnum neg edges size = {}'.format(len(cand_edges),num_neg_edges))
        idxes = np.random.choice(range(len(cand_edges)),size,replace=False)
        neg_edges = [cand_edges[idx] for idx in idxes]
        # neg_edges =[]
        # count = 0
        # inf_loop = 5000000
        # num_it = 0
        # while count < num_neg_edges:
        #     candidtate = tuple(np.random.choice(nodes,2,replace=False))
        #     if not G.has_edge(*candidtate) and candidtate not in neg_edges:
        #         neg_edges.append(candidtate)
        #         count += 1
        #     num_it += 1
        #     if num_it > inf_loop:
        #         print('FUCK!?!?!?')
        #         exit()
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

        self.train_neg_edges['between'].update(d)
        # self.train_neg_edges['between'].setdefault(communities,[]).extend(neg_between_edges(edges,nodes1,nodes2))

    def create_train_neg_between_edges(self):
        community_dict = self.community_dict
        num_cores = min(self.num_cores,len(self.between_edges_dict.keys()))
        sliced_between_edges_dict = self.slice_dict(self.between_edges_dict,num_cores * self.multiplier)
        Parallel(n_jobs=self.num_cores)(delayed(self.parallel_train_neg_between)(between_edges_dict_chunk,community_dict)
                                        for between_edges_dict_chunk in sliced_between_edges_dict)

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

        self.train_neg_edges['within'].update(d)
        # self.train_neg_edges['within'].setdefault(label,[]).extend(self.create_test_neg_within_edges(nodes,N))

    def create_train_neg_within_edges(self):
        # train_neg_edges = self.train_neg_edges
        # community_dict = self.community_dict
        # G = self.G
        # for label,nodes in community_dict.items():
        #     N = len(list(G.subgraph(nodes).edges()))
        #     train_neg_edges['within'].setdefault(label,[]).extend(self.create_test_neg_within_edges(nodes,N))

        num_chunks = min(self.num_cores * 5 ,len(self.community_dict.keys()))
        sliced_community_dict = self.slice_dict(self.community_dict,num_chunks)
        Parallel(n_jobs=self.num_cores)(delayed(self.parallel_train_neg_within)(community_dict_chunk)
                                        for community_dict_chunk in sliced_community_dict)


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
        # print('before between train')
        # for c,nodes in self.community_dict.items():
        #     print('is community {} connected?\n{}'.format(c,nx.is_connected(self.G.subgraph(nodes))))
        self.create_train_neg_between_edges()
        # print('after between train')
        # for c, nodes in self.community_dict.items():
        #     print('is community {} connected?\n{}'.format(c, nx.is_connected(self.G.subgraph(nodes))))
        self.train_neg_edges['between'] = dict(self.train_neg_edges['between'])
        print('it took {:.3f} seconds to create train neg betweeen edges'.format(time()-start))
        start = time()
        # print('before within train')
        # for c, nodes in self.community_dict.items():
        #     print('is community {} connected?\n{}'.format(c, nx.is_connected(self.G.subgraph(nodes))))
        self.create_train_neg_within_edges()
        # print('after within train')
        # for c, nodes in self.community_dict.items():
        #     print('is community {} connected?\n{}'.format(c, nx.is_connected(self.G.subgraph(nodes))))
        # exit()
        self.train_neg_edges['within'] = dict(self.train_neg_edges['within'])

        self.train_neg_edges = dict(self.train_neg_edges)
        print('it took {:.3f} seconds to create train neg within edges'.format(time()-start))
        start = time()
        train_pos_edges = {'within': self.get_within_community_edges(), 'between':self.get_between_edges_dict()}
        obj = {'pos': train_pos_edges,'neg': self.train_neg_edges}


        ############ TO  DELETE ###############
        # print('train between pos links:\n{}\ntrain between neg links:\n{}'.format(train_pos_edges['between'],self.train_neg_edges['between']))
        # print('train within pos links:\n{}\ntrain within neg links:\n{}'.format(train_pos_edges['within'],self.train_neg_edges['within']))
        # exit()
        ############ TO  DELETE ###############

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
