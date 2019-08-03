import numpy as np
from file2obj import save_to_gzip


# Choice of binary operators ο for learning edge features
Hadamard = lambda u,v: np.multiply(u , v)
Average = lambda u,v: np.mean(u,v)
Weighted_L1 = lambda u,v: np.abs(u - v)
Weighted_L2 = lambda u,v:np.sqare(u - v)

class Edge2vec(object):

    def __init__(self,vectors,is_pos = False):
        self.vectors = vectors
        self.is_pos= is_pos

    def adjust_n2vdict(self):
        vectors = self.vectors
        new_dict = {}
        for (c,u),val in vectors.items():
            new_dict.setdefault(u,[]).extend([val])
        new_dict = {u:np.mean(val,axis=0) for u,val in new_dict.items()}
        self.vectors = new_dict

    def max_n2v(self,G,community_dict):
        vectors = self.vectors
        new_dict = {}
        val_dict = {}
        for c,nodes in community_dict.items():
            G_sub = G.subgraph(nodes)
            denom = len(G_sub.edges())
            for node in nodes:
                new_val = len(list(G_sub.neighbors(node))) / denom
                val = val_dict.get(node,(c,0.0))[1]
                if val < new_val:
                    val_dict[node] = (c,new_val)
        for u,(c,_) in val_dict.items():
            new_dict[str(u)] = vectors[(c,str(u))]

        self.vectors = new_dict


    def edge2vec(self,edges,operator,vec_dim,is_pos = False):
        vectors = self.vectors
        num_edges = len(edges)
        edge_vectors = np.zeros((num_edges,vec_dim))
        edge_labels = np.zeros((num_edges,1),dtype = np.int)
        # count = 0
        for i,(node1,node2) in enumerate(edges):
            edge_vectors[i,:] += operator(vectors[str(node1)],vectors[str(node2)])
        if is_pos:
            edge_labels += 1

        new_edges = np.hstack((edge_vectors,edge_labels))
        return new_edges

    # def edge2vec_community(self,edge_tuples,operator,vec_dim,is_pos = False):
    #
    #     vectors = self.vectors
    #     num_edges = np.sum(len(edges) for c,edges in edge_tuples)
    #     edge_vectors = np.zeros((num_edges,vec_dim))
    #     edge_labels = np.zeros((num_edges,1),dtype = np.int)
    #     # count = 0
    #     for i,edge in enumerate(edge_tuples):
    #         c1, node1 = edge[0]
    #         c2, node2 = edge[1]
    #         edge_vectors[i,:] += operator(vectors[(c1,str(node1))],vectors[(c2,str(node2))])
    #     if is_pos:
    #         edge_labels += 1
    #
    #     new_edges = np.hstack((edge_vectors,edge_labels))
    #     return new_edges

'''
create a dictionary of positive and negative edges (for train and test)
before using the function create_edge2vec_dataset
'''


merge_edges = lambda  pos_edges,neg_edges : np.vstack((pos_edges,neg_edges))

def create_pos_neg_dict(pos_edges,neg_edges):
    edges = pos_edges.copy()
    edges.update(neg_edges)
    return edges

'''
save training/test to file
last columns represent y (X is the rest)
'''
def save_dataset(X,y,filename):
    # y = y.reshape((y.shape[0],1))
    dataset = np.hstack((X,y))
    save_to_gzip(dataset,filename)
    # fin = open(filename,'rb')
    # np.save(fin,dataset)
    # fin.close()