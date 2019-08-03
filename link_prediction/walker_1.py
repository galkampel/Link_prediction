import random
import numpy as np
from joblib import Parallel,delayed
from multiprocessing import cpu_count
from file2obj import save_to_gzip
from itertools import chain
from time import time
import gc


random.seed(1)
np.random.seed(1)
sliced_el = lambda lst,num_chunks: list(map(list, list(lst[i::num_chunks] for i in range(num_chunks))))
expand_node = lambda node,c : '{}_{}'.format(node,c)
node2orig = lambda node:  node.split('_')
get_node = lambda node_c: node_c.split('_')[0]


def deepwalk_walk_wrapper(class_instance, walk_length, start_node):
    class_instance.deepwalk_walk(walk_length, start_node)

def get_random_label(G,u_c):
    u = get_node(u_c)
    labels = G.nodes[u]['label']
    return labels[np.random.randint(len(labels))]

class BasicWalker:
    def __init__(self, G, workers):
        self.G = G.G
        # self.node_size = G.node_size
        # self.look_up_dict = G.look_up_dict

    def deepwalk_walk(self, walk_length, start_node):
        '''
        Simulate a random walk starting from start node.
        '''
        G = self.G
        # look_up_dict = self.look_up_dict
        # node_size = self.node_size

        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = list(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                walk.append(random.choice(cur_nbrs))
            else:
                break
        return walk

    def simulate_walks(self, num_walks, walk_length):
        '''
        Repeatedly simulate random walks from each node.
        '''

        G = self.G
        walks = []
        nodes = list(G.nodes())
        print('Walk iteration:')
        for walk_iter in range(num_walks):
            # pool = multiprocessing.Pool(processes = 4)
            print(str(walk_iter+1), '/', str(num_walks))
            random.shuffle(nodes)
            for node in nodes:
                # walks.append(pool.apply_async(deepwalk_walk_wrapper, (self, walk_length, node, )))
                walks.append(self.deepwalk_walk(walk_length=walk_length, start_node=node))
            # pool.close()
            # pool.join()
        # print len(walks)
        return walks


class Walker:
    def __init__(self, G, p, q, community_dict,workers):
        self.G = G
        self.p = p
        self.q = q
        self.community_dict = community_dict
        # self.node_size = G.node_size
        # self.look_up_dict = G.look_up_dict

    def set_alias_nodes(self,alias_nodes):
        self.alias_nodes = alias_nodes

    def set_alias_edges(self,alias_edges):
        self.alias_edges = alias_edges

    def node2vec_walk(self, walk_length, start_node):
        '''
        Simulate a random walk starting from start node.
        '''
        G = self.G

        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges
        # look_up_dict = self.look_up_dict
        # node_size = self.node_size

        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = list(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    walk.append(cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
                else:
                    prev = walk[-2]
                    pos = (prev, cur)
                    next = cur_nbrs[alias_draw(alias_edges[pos][0],
                        alias_edges[pos][1])]
                    walk.append(next)
            else:
                break

        return walk

    def get_all_nodes(self):
        community_dict = self.community_dict
        N = sum(len(nodes) for nodes in community_dict.values())
        all_nodes = [None] * N
        t = 0
        for c,nodes in community_dict.items():
            for node in nodes:
                all_nodes[t] = node
                t += 1
        return all_nodes,N

    def parallel_walk(self,iters,walk_length):#,walks):
        start = time()
        nodes,N = self.get_all_nodes()
        walks = [None] * (N * iters)
        print('nodes length = {}'.format(N))
        for i in range(iters):
            # start = time()
            random.shuffle(nodes)
            walks[i*N:(i+1)*N] = [self.node2vec_walk(walk_length=walk_length, start_node=node) for node in nodes]
            # print('it took {:.3f} seconds to finish iters'.format(time()-start))
        print('it took {:.3f} seconds to create walk for all nodes'.format(time()-start))
        return walks

    def simulate_walks(self, num_walks, walk_length,num_cores = 3 ): #is_community_level
        '''
        Repeatedly simulate random walks from each node.
        '''
        # iters = [num_walks // num_cores] * num_cores
        # if num_walks % num_cores != 0:
        #     resid = num_walks % num_cores
        #     idx = 0
        #     while resid > 0:
        #         iters[idx] += 1
        #         resid -= 1
        #         idx = (idx + 1) % (num_cores - 1)
        # print('iters:\n{}'.format(iters))
        # num_cores = cpu_count()
        print('one-one-two')
        walks = Parallel(n_jobs=num_cores)(delayed(self.parallel_walk)(par_iters,walk_length)#,walks) #
                               for par_iters in range(num_walks))
        walks = list(chain.from_iterable(walks))
        # print('it took {:.3f} seconds to create walks'.format(time()-start))
        # exit()
        return walks

    def get_alias_edge(self, src, dst):
        '''
        Get the alias edge setup lists for a given edge.
        '''
        G = self.G
        p = self.p
        q = self.q

        unnormalized_probs = []
        for dst_nbr in G.neighbors(dst):
            if dst_nbr == src:
                unnormalized_probs.append(G[dst][dst_nbr]['weight']/p)
            elif G.has_edge(dst_nbr, src):
                unnormalized_probs.append(G[dst][dst_nbr]['weight'])
            else:
                unnormalized_probs.append(G[dst][dst_nbr]['weight']/q)
        norm_const = sum(unnormalized_probs)
        normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]

        return alias_setup(normalized_probs)

    def parallel_nodes(self,nodes):
        alias_nodes = {}
        G = self.G
        for node in nodes:
            unnormalized_probs = [G[node][nbr]['weight'] for nbr in G.neighbors(node)]
            norm_const = sum(unnormalized_probs)
            normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]
            alias_nodes[node] = alias_setup(normalized_probs)
        return alias_nodes
    def parallel_edges(self,edges):
        alias_edges = {}
        start = time()
        for edge in edges:
            alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
        print('it took {:.3f} seconds to create edges chunks'.format(time() - start))
        return alias_edges

    def preprocess_transition_probs(self,multiplier,n_jobs=cpu_count()):
        '''
        Preprocessing of transition probabilities for guiding the random walks.
        Need to be parallelized and saved (calculated only once)
        '''
        G = self.G

        start = time()
        num_chunks_edges = int(n_jobs *multiplier ) #multiplier * 2.5
        num_chunks_nodes = int(n_jobs / 6)
        print('num node chunks = {}'.format(num_chunks_nodes))
        sliced_nodes = sliced_el(list(G.nodes()),num_chunks_nodes)
        alias_nodes_lst = Parallel(n_jobs=num_chunks_nodes)(
            delayed(self.parallel_nodes)(nodes) for nodes in sliced_nodes)
        print('it took {:.3f} seconds to create alias nodes'.format(time()-start))
        alias_nodes = {}

        sliced_edges = sliced_el(list(G.edges()),num_chunks_edges)
        print('num edge-chunks = {}'.format(num_chunks_edges))
        alias_edges_lst = Parallel(n_jobs=n_jobs)(
            delayed(self.parallel_edges)(edges) for edges in sliced_edges)
        alias_edges = {}
        # print('n_jobs = {}\nnum chunks  = {}'.format(n_jobs,num_chunks))
        for i in range(num_chunks_nodes):
            alias_nodes.update(alias_nodes_lst[i])
        for i in range(num_chunks_edges):
            alias_edges.update(alias_edges_lst[i])


        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges
        if multiplier > 1:
            save_to_gzip(alias_nodes,'alias_nodes_p={}_q={}'.format(self.p,self.q))
            save_to_gzip(alias_edges,'alias_edges_p={}_q={}'.format(self.p,self.q))
        return


class CommunityWalker:
    def __init__(self, G, community_dict,params,rho = 0.3):
        self.G = G
        self.community_dict = community_dict
        self.params = params
        self.rho = params["rho"]


    def node2vec_walk(self,start_node,to_expand):
        '''
        Simulate a random walk starting from start node.
        '''
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges
        G = self.G

        walk = []
        rho = self.rho
        v_cur,start_c = node2orig(start_node)
        get_prev = lambda el: el
        node2walk = lambda u, c: u

        if to_expand:
            get_prev = get_node
            node2walk = expand_node

        v_c = start_c
        walk.append(node2walk(v_cur, v_c))
        cur_nbrs = None
        while len(walk) < self.params["l"]:
            if len(walk) == 1:
                cur_nbrs = list(G.neighbors(v_cur))
                v_cur = cur_nbrs[alias_draw(alias_nodes[(v_cur,v_c)][0], alias_nodes[(v_cur,v_c)][1])]
                v_Cs = list(G.nodes[v_cur]['label'])
                vals = [1 - rho if v_c == c else rho for c in v_Cs]
                norm_const = sum(vals)
                probs = [val / norm_const for val in vals]
                v_c = np.random.choice(v_Cs,p = probs)
            elif len(cur_nbrs) > 0 :
                cur_nbrs = list(G.neighbors(v_cur))
                prev = get_prev(walk[-2])
                pos = (prev, v_cur,v_c)
                v_cur = cur_nbrs[alias_draw(alias_edges[pos][0], alias_edges[pos][1])]
                v_Cs = list(G.nodes[v_cur]['label'])
                vals = [1 - rho if v_c == c else rho for c in v_Cs]
                norm_const = sum(vals)
                probs = [val / norm_const for val in vals]
                v_c = np.random.choice(v_Cs,p = probs)
            else:     #if we have reached to a dead end
                break

            walk.append(node2walk(v_cur,v_c))

        return walk


    def get_all_nodes(self):
        community_dict = self.community_dict
        all_nodes = []
        for c,nodes in community_dict.items():
            for node in nodes:
                all_nodes.append(expand_node(node,c))
        return all_nodes,len(all_nodes)

    def parallel_walk(self, iters, to_expand):
        # start = time()
        nodes, N = self.get_all_nodes()
        walks = [None] * (N * iters)
        for i in range(iters):
            random.shuffle(nodes)
            walks[i * N: (i + 1) * N] = [self.node2vec_walk(start_node=node, to_expand=to_expand) for node in nodes]  # expand_node(get_node(node),get_random_label(self.G,node))
        # print('it took {:.3f} seconds to create walk for all nodes'.format(time()-start))
        return walks


    def simulate_walks(self, num_walks, to_expand, num_cores=3):
        '''
        Repeatedly simulate random walks from each node.
        '''
        iters = [num_walks // num_cores] * num_cores
        if num_walks % num_cores != 0:
            resid = num_walks % num_cores
            idx = 0
            while resid > 0:
                iters[idx] += 1
                resid -= 1
                idx = (idx + 1) % (num_cores - 1)
        # print('iters:\n{}'.format(iters))
        # num_cores = cpu_count()
        walks = Parallel(n_jobs=num_cores)(delayed(self.parallel_walk)(par_iters, to_expand)
                                           for par_iters in iters)
        walks = list(chain.from_iterable(walks))
        # print('it took {:.3f} seconds to create walks'.format(time()-start))
        return walks

    # def simulate_walks(self, num_walks, to_expand):
    #     '''
    #     Repeatedly simulate random walks from each node.
    #     '''
    #     nodes, N = self.get_all_nodes()
    #     walks = [None] * (N * num_walks)
    #     for i in range(num_walks):
    #         random.shuffle(nodes)
    #         walks[i * N: (i + 1) * N] = [self.node2vec_walk(start_node=node, to_expand=to_expand) for node in nodes]  # expand_node(get_node(node),get_random_label(self.G,node))
    #         print('Finished iteration number {}'.format(i))
    #     # print('it took {:.3f} seconds to create walks'.format(time()-start))
    #     return walks


    def get_alias_edge(self,alias_edges,src, dst):
        '''
        Get the alias edge setup lists for a given edge.
        '''
        G = self.G
        p = self.params["p"]
        q = self.params["q"]
        rho = self.rho
        for c in G.nodes[dst]['label']:
            unnormalized_probs = []
            for dst_nbr in G.neighbors(dst):
                if dst_nbr == src:
                    unnormalized_probs.append((1-rho) * G[dst][dst_nbr]['weight'] /p
                                              if c in G.nodes[dst_nbr]['label'] else
                                               rho * G[dst][dst_nbr]['weight'] / p )
                elif G.has_edge(dst_nbr, src):
                    unnormalized_probs.append((1-rho) *G[dst][dst_nbr]['weight']
                                              if c in G.nodes[dst_nbr]['label'] else
                                               rho * G[dst][dst_nbr]['weight'] )
                else:
                    unnormalized_probs.append((1-rho) *G[dst][dst_nbr]['weight'] /q
                                              if c in G.nodes[dst_nbr]['label'] else
                                               rho * G[dst][dst_nbr]['weight'] / q )
            norm_const = sum(unnormalized_probs)
            normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]
            alias_edges[(src, dst,c)] = alias_setup(normalized_probs)
        return

    def parallel_nodes(self,nodes):
        alias_nodes = {}
        G = self.G
        rho =self.rho
        for node in nodes:
            for c in G.nodes[node]['label']:
                unnormalized_probs = [(1-rho)* G[node][nbr]['weight'] if c in G.nodes[nbr]['label']
                                      else rho* G[node][nbr]['weight'] for nbr in G.neighbors(node)]
                norm_const = sum(unnormalized_probs)
                normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]
                alias_nodes[node,c] = alias_setup(normalized_probs)
        # gc.collect()
        return alias_nodes

    def parallel_edges(self,edges):
        alias_edges = {}
        start = time()
        for edge in edges:
            self.get_alias_edge(alias_edges,edge[0], edge[1])
        # gc.collect()
        print('it took {:.3f} seconds to create edges chunks'.format(time()-start))
        return alias_edges

    def preprocess_transition_probs(self,multiplier,n_jobs=cpu_count()):
        '''
        Preprocessing of transition probabilities for guiding the random walks.
        Need to be parallelized and saved (calculated only once)
        '''

        G = self.G
        start = time()
        num_chunks_edges = int(n_jobs * multiplier * 1.2)  # multiplier * 2.5
        num_chunks_nodes = int(n_jobs / 6)
        sliced_nodes = sliced_el(list(G.nodes()),num_chunks_nodes)
        alias_nodes_lst = Parallel(n_jobs=num_chunks_nodes)(
            delayed(self.parallel_nodes)(nodes) for nodes in sliced_nodes)

        print('it takes {:.3f} to create alias nodes dict'.format(time() - start))
        sliced_edges = sliced_el(list(G.edges()),num_chunks_edges )
        alias_edges_lst = Parallel(n_jobs=n_jobs)(
            delayed(self.parallel_edges)(edges) for edges in sliced_edges)

        # print('There are {} keys in alias edges'.format(sum(len(el) for el in alias_edges_lst)))
        alias_nodes = {}
        alias_edges = {}

        for i in range(num_chunks_nodes):
            alias_nodes.update(alias_nodes_lst[i])
        for i in range(num_chunks_edges):
            alias_edges.update(alias_edges_lst[i])

        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges
        if multiplier > 1:
            params = self.params
            p,q = params["p"],params["q"]
            save_to_gzip(alias_nodes,'alias_nodes_p={}_q={}_rho={}'.format(p,q,self.rho))
            save_to_gzip(alias_edges,'alias_edges_p={}_q={}_rho={}'.format(p,q,self.rho))
        return


def alias_setup(probs):
    '''
    Compute utility lists for non-uniform sampling from discrete distributions.
    Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    for details
    '''
    K = len(probs)
    a = sum(probs) / K
    q = np.zeros(K, dtype=np.float32)
    J = np.arange(K, dtype=np.int32)

    smaller = []
    larger = []
    for i, prob in enumerate(probs):
        q[i] = (i+1) * a
        if prob <= a:
            smaller.append((i,prob))
        else:
            larger.append((i,prob))

    while len(smaller) > 0 and len(larger) > 0:
        s, prob_s = smaller.pop()
        l, prob_l = larger.pop()

        if prob_s < a:
            J[s] = l
            q[s] = a * s+ prob_s
            prob_l = prob_l - (a - prob_s)

        if prob_l <= a:
            smaller.append((l,prob_l))
        else:
            larger.append((l,prob_l))

    return J, q

def alias_draw(J, q):
    '''
    Draw sample from a non-uniform discrete distribution using alias sampling.
    '''
    K = len(J)
    u = np.random.rand()
    idx = int(np.floor(u *K))
    if u < q[idx]:
        return idx
    else:
        return J[idx]