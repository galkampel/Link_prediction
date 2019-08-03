
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



    def parallel_within_edges(self,communities_dict_chunk,test_neg_edges,test_pos_edges):
        test_neg_within = {}
        test_pos_within = {}
        for label,nodes in communities_dict_chunk.items():

            G_sub = self.G.subgraph(nodes)
            num_edges = int(self.frac_within * len(G_sub.edges()))
            neg_edges = self.create_test_neg_within_edges(nodes,num_edges)
            num_edges = len(neg_edges)
            test_neg_within.setdefault(label,[]).extend(neg_edges)
            #get positive within
            # print('G sub has {} nodes and {} edges'.format(len(G_sub.nodes()),len(G_sub.edges())))
            edges = self.create_positive_edge_within(G_sub,num_edges)
            test_pos_within.setdefault(label,[]).extend(edges)

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
            tmp_dict_neg[communities] = self.create_neg_between_edges(nodes1,nodes2,num_edges)
            tmp_dict_pos[communities] = self.create_positive_edge_between(edges,num_edges)

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
    def set_new_G(self,filename):
        # test_neg_edges = {'within':{},'between':{}}
        # test_pos_edges ={'within':{},'between':{}}
        G = self.G
        community_dict = self.community_dict
        manager = Manager()
        test_neg_edges = manager.dict()
        test_neg_edges['within'] = manager.dict()
        test_neg_edges['between'] = manager.dict()
        test_pos_edges = manager.dict()
        test_pos_edges['within'] = manager.dict()
        test_pos_edges['between'] = manager.dict()
        # start = time()
        num_cores = min(self.num_cores,len(community_dict.keys()))
        sliced_community_dict = self.slice_dict(community_dict,num_cores)
        Parallel(n_jobs=self.num_cores)(delayed(self.parallel_within_edges)(communities_dict_chunk,test_neg_edges,test_pos_edges)
                                   for communities_dict_chunk in sliced_community_dict)
        # print('It took {:.3f} seconds to create neg within edges'.format(time()-start))
        test_pos_edges['within'] = dict(test_pos_edges['within'])
        test_neg_edges['within'] = dict(test_neg_edges['within'])
        # print('test pos within non-unique equlas to test pos within unique?\n{}'.format(len(list(chain(*test_pos_edges['within'].values()))) == len(set(list(chain(*test_pos_edges['within'].values()))))))

        good_dict,problematic_dict = {},{}
        for c in test_pos_edges['within'].keys():
            for edge in test_pos_edges['within'][c]:
                if len(set(G.nodes[edge[0]]['label']).intersection(set(G.nodes[edge[1]]['label']))) == 1:
                    good_dict.setdefault(c,[]).append(edge)
                else:
                    problematic_dict.setdefault(c,[]).append(edge)

        print('there are {} problematic communities and {} problematic links'.format(len(problematic_dict),len(list(chain(*problematic_dict.values())))))
        print('good dict has {} communities'.format(len(good_dict)))
        G.remove_edges_from(list(chain(*good_dict.values())))
        num_problematic_edges = 0
        G_subs = {c: G.subgraph(nodes).copy() for c, nodes in community_dict.items()}
        good_links = []
        for c_problem in problematic_dict.keys():
            for edge in problematic_dict[c_problem]:
                rel_communities = set(G.nodes[edge[0]]['label']).intersection(set(G.nodes[edge[1]]['label']))
                rollback_c = []
                is_violated = False
                for c in rel_communities:
                    if edge not in list(G_subs[c].edges()) :
                        G_subs[c].add_edge(*edge)
                    rollback_c.append(c)
                    G_subs[c].remove_edge(*edge)
                violations = [1 if not nx.is_connected(G_subs[c_check]) else 0 for c_check in community_dict.keys()]
                if np.sum(violations) > 0:
                    is_violated = True
                    for c2 in rollback_c:
                        G_subs[c2].add_edge(*edge)
                    num_problematic_edges += 1
                        # break
                if not is_violated:
                    good_links.append(edge)
                    good_dict.setdefault(c_problem,[]).append(edge)
        print('{} pos links were removed'.format(num_problematic_edges))
        print('There are {} pos links before removal'.format(len(list(chain(*test_pos_edges['within'].values())))))
        if num_problematic_edges > 0:
            test_pos_edges['within'] = good_dict
        print('There are {} pos links after removal'.format(len(list(chain(*test_pos_edges['within'].values())))))
        print('test_pos_edges[within] has {} communities'.format(len(test_pos_edges['within'])))
        G.remove_edges_from(good_links)
        between_edges_dict = self.between_edges_dict
        # start = time()
        num_cores = min(self.num_cores,len(between_edges_dict.keys()))
        sliced_between_edges_dict = self.slice_dict(between_edges_dict,num_cores)

        Parallel(n_jobs=self.num_cores)(delayed(self.parallel_between_edges)(communities_chunk,community_dict,test_neg_edges,test_pos_edges)
                                    for communities_chunk in sliced_between_edges_dict)

        test_pos_edges['between'] = dict(test_pos_edges['between'])
        self.update_between_edges_dict(test_pos_edges['between'])
        test_neg_edges['between'] = dict(test_neg_edges['between'])
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
        # print('range length:\n{}'.format(len(range_edges)))
        G_copy = G_sub.copy()

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
        sliced_between_edges_dict = self.slice_dict(self.between_edges_dict,num_cores)
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
        num_cores = min(self.num_cores,len(self.community_dict.keys()))
        sliced_community_dict = self.slice_dict(self.community_dict,num_cores)
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
