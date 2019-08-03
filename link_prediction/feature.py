import networkx as nx
import numpy as np
from joblib import Parallel,delayed
from multiprocessing import cpu_count
# from itertools import product,chain

class Features(object):
    def __init__(self,G,num_features = 9): #num_orig_classes,num_neg_classes,is_labeled = False
        self.G = G
        # self.num_orig_classes = num_orig_classes
        # self.num_neg_classes = num_neg_classes
        # self.is_labeled = is_labeled
## dictionary of neighbors
        self.neighbors = {node:list(G.neighbors(node))  for node in G.nodes()}
        # self.neighbors_neighbors = {u : set(list(chain(*[self.neighbors[u1] for u1 in self.neighbors[u]]))) for u in G.nodes()}
        self.num_features = num_features


    def parallelized_vector(self,edges,label):
        num_edges = len(edges)
        edge_vectors = np.zeros((num_edges,self.num_features))
        edge_labels = np.zeros((num_edges,1))
        neighbors = self.neighbors
        common_idx, total_idx, jaccard_idx,adamic_idx,preferential_idx,\
        rokhlin_idx,friends_idx, resource_idx, LHN_idx  = range(self.num_features)

        count = 0
        for i,(u,v) in enumerate(edges):
            common_friends = set(neighbors[u]).intersection(set(neighbors[v]))
            edge_vectors[i+count,common_idx] = len(common_friends)
            total_friends = set(neighbors[u]).union(set(neighbors[v]))
            edge_vectors[i+count,total_idx] = len(total_friends)
            edge_vectors[i+count,jaccard_idx] = self.jaccard(edge_vectors,i,common_idx,total_idx)
            edge_vectors[i+count,adamic_idx] = self.Adamic_Adar(common_friends)
            edge_vectors[i+count,preferential_idx] = self.preferential_measure(u,v)
            edge_vectors[i+count,rokhlin_idx] = len(total_friends.difference(common_friends))
            edge_vectors[i+count,friends_idx] = self.friends_measure(u,v)
            # edge_vectors[i+count,friends_idx:friends_friends_idx+1] = self.friends_measure_and_friends_edges(u,v)
            edge_vectors[i+count,resource_idx] = self.resource_allocation(common_friends)
            edge_vectors[i+count,LHN_idx] = self.LHN(common_friends,u,v)
            edge_labels[i+count,0] = label

        new_edges = np.hstack((edge_vectors,edge_labels))
        # par_list.append(new_edges)
        return new_edges
##### edges: dict (key-label,value-list of nodes)
# 4 dict edges (train_pos,train_neg,test_pos,test_neg)
    def create_dataset(self,edges,label):

        num_chunks = cpu_count()
        sliced_edges = list(map(list,list(edges[i::num_chunks] for i in range(num_chunks))))
        par_list = Parallel(n_jobs=num_chunks)(delayed(self.parallelized_vector)(chunk,label) for chunk in sliced_edges)
        count = 0
        num_edges = len(edges)
        data = np.zeros((num_edges,self.num_features+1))
        for arr in par_list:
            N = arr.shape[0]
            data[count:count+N,:] = arr
            count += N
        # start = time()
        # data = np.reshape(np.array(par_list),(-1,par_list[0].shape[1]))
        # data = np.vstack(list(par_list))
        # print('it takes {:.3f} seconds to create data'.format(time()-start))
        # print(data.shape)
        return data




    def jaccard(self,vec,i,common_idx,total_idx):
        if vec[i,common_idx] == 0:
            return 0
        return vec[i,common_idx] / vec[i,total_idx]

    def Adamic_Adar(self,common_friends):
        neighbors = self.neighbors
        res = 0
        if len(common_friends) == 0:
            return res
        for z in common_friends:
            res +=  1.0 / np.log(len(neighbors[z])+1)
        return res

    def preferential_measure(self,u,v):
        return len(self.neighbors[u]) * len(self.neighbors[v])


    def friends_measure(self,u,v):
        G = self.G
        neighbors = self.neighbors
        res = 0
        # res = np.sum([1 for x in neighbors[u] for y in neighbors[v]  if G.has_edge(x,y) or x == y])
        for x in neighbors[u]:
            for y in neighbors[v]:
                if G.has_edge(x,y) or x == y:
                    res += 1
        return res

    # def friends_measure_and_friends_edges(self,u,v):
    #     G = self.G
    #     neighbors = self.neighbors
        res_friends1 = 0
        # start = time()
        # prod = list(product(neighbors[u],neighbors[v]))
        # res_friends1 = np.sum(list(map(lambda el: 1 if G.has_edge(el[0],el[1]) or el[0] == el[1] else 0,product(neighbors[u],neighbors[v]))))
        # u2s = []
        # for u1 in neighbors[u]:
        #     u2s.append(neighbors[u1])
        # v2s = []
        # for v1 in neighbors[v]:
        #     v2s.append(neighbors[v1])
        # prod = list(product(list(chain(*[neighbors[u1] for u1 in neighbors[u]])),list(chain(*[neighbors[v1] for v1 in neighbors[v]]))))
        # res_friends2 = np.sum(list(map(lambda el: 1 if G.has_edge(el[0],el[1]) or el[0] == el[1] else 0,
        #                             product(self.neighbors_neighbors[u],self.neighbors_neighbors[v]))))
        # for u1 in neighbors[u]:
        #     for v1 in neighbors[v]:
        #         if G.has_edge(u1,v1) or u1 == v1:
        #             res_friends1 += 1
        #
        # res_friends2 = len(G.subgraph(neighbors[u] + neighbors[v]).edges())
        # if res_friends2 == 0:
        #     print('What the ****?!!??')
        #     print('u = {}\tN(u) = {}\nv = {}\tN(v) = {}'.format(u,neighbors[u],v,neighbors[v]))
        #     print('num common edges: {}'.format(len(G.subgraph(neighbors[u] + neighbors[v]).edges())))
        # for u1 in neighbors[u]:
        #     for v1 in neighbors[v]:
        #         if G.has_edge(u1,v1) or u1 == v1:
        #             res_friends1 += 1
        #         prod = list(product(neighbors[u1],neighbors[v1]))
        # tmp_res = np.sum(list(map(lambda el: 1 if G.has_edge(el[0],el[1]) or el[0] == el[1] else 0,prod)))
        # res_friends2 += tmp_res
                # for u2 in neighbors[u1]:
                #     for v2 in neighbors[v1]:
                #         if G.has_edge(u2,v2) or u2 == v2:
                #             res_friends2 += 1
        # print('it took {:.3f} seconds to create 2 elemets'.format(time()-start))
        # return res_friends1,res_friends2


    def resource_allocation(self,common_friends):
        neighbors = self.neighbors
        res = 0
        for z in common_friends:
            res += len(neighbors[z])
        if res == 0:
            return res
        res = res ** -1
        return res

# Leicht-Holme-Newman
    def LHN(self,common_friends,u,v):
        neighbors = self.neighbors
        return len(common_friends) / (len(neighbors[u]) * len(neighbors[v]))



