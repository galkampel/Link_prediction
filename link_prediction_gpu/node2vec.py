
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from walker import *

import networkx as nx
from sklearn.cluster import KMeans,Birch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout
from time import time

# fig_size = plt.rcParams["figure.figsize"]
# fig_size[0] = 9.6#8
# fig_size[1] = 7.6#6
class Node2vec(object):

    def __init__(self, G, params,community_dict,is_community_level=False,
                 to_expand = False,to_intertwine = False,frac_between = 0.5,dw=False, **kwargs): #is_community_level=True,

        kwargs["workers"] = kwargs.get("workers", 1)
        if dw:
            # kwargs["hs"] = 1
            p = 1.0
            q = 1.0

        self.is_community_level = is_community_level
        self.l = params["l"]
        self.r = params["r"]
        self.d = kwargs["size"]
        self.p = params["p"]
        self.q = params["q"]
        self.k = kwargs["window"]

        self.G = G

        sentences = None
        if dw:
            self.walker = BasicWalker(G, workers=kwargs["workers"])
        elif is_community_level and  to_intertwine:
            start = time()
            params["rho"] = 0.7
            self.walker = CommunityWalker(G,community_dict,params)
            self.walker.preprocess_transition_probs() #kwargs["workers"]
            num_walks = int(self.r * frac_between)
            sentences1 = self.walker.simulate_walks(num_walks=num_walks,to_expand = to_expand)
            params["rho"] = 0.3
            num_walks = int(self.r * (1 - frac_between) )
            self.walker = CommunityWalker(G,community_dict,params)
            self.walker.preprocess_transition_probs() #kwargs["workers"]
            sentences2 = self.walker.simulate_walks(num_walks=num_walks,to_expand = to_expand)
            sentences = sentences1 + sentences2
            print('it took {:.3f} seconds to create transition probs'.format(time()-start))
        elif is_community_level and not to_intertwine:
            self.walker = CommunityWalker(G,community_dict,params)
        else:
            self.walker = Walker(G, p=self.p, q=self.q,community_dict=community_dict, workers=kwargs["workers"])


        if is_community_level and not to_intertwine:
            start = time()
            self.walker.preprocess_transition_probs() #kwargs["workers"]
            print('it took {:.3f} seconds to create transition probs'.format(time()-start))
            start = time()
            sentences = self.walker.simulate_walks(num_walks=self.r,to_expand = to_expand)
            print('it took {:.3f} seconds to simulate walks'.format(time()-start))
        elif not is_community_level:
            start = time()
            self.walker.preprocess_transition_probs() #kwargs["workers"]
            print('it took {:.3f} seconds to create transition probs'.format(time()-start))
            sentences = self.walker.simulate_walks(num_walks=self.r, walk_length=self.l)#is_community_level = self.is_community_level
        kwargs["sentences"] = sentences
        kwargs["min_count"] = kwargs.get("min_count", 0)
        kwargs["size"] = kwargs.get("size", self.d)
        kwargs["sg"] = 1
        kwargs["window"] = kwargs.get("window",self.k)
        self.size = kwargs["size"]
        #improving node2vec by increasing #epochs (iter),
        kwargs["compute_loss"] = True

        kwargs["iter"] = 5 #100
        lrs = [0.3] #0.025


        # kwargs["iter"] = 20
        # lrs = [0.4,0.3,0.25,0.1,0.025,0.001,0.0001]

        # print ("Learning representation...")
        # print('kwargs:\n{}'.format(kwargs))
        # print('num workers: {}'.format(kwargs["workers"]))

        word2vec = None
        self.vectors = {}

        if not is_community_level:
            params = {'p':self.p,"q":self.q,"d": self.d,"r":self.r,"l":self.l,"k":self.k}
            for lr in lrs:
                kwargs["alpha"] = lr
                epoch_logger = EpochLogger(kwargs["iter"],lr,params,is_community_level)
                kwargs["callbacks"] = [epoch_logger]
                word2vec = Word2Vec(**kwargs)
                for word in G.nodes():
                    self.vectors[word] = word2vec[word]
        else:
            for lr in lrs:
                kwargs["alpha"] = lr
                epoch_logger = EpochLogger(kwargs["iter"], lr, params,is_community_level)
                kwargs["callbacks"] = [epoch_logger]
                word2vec = Word2Vec(**kwargs)
                if to_expand:
                    for c,nodes in community_dict.items():
                        for node in nodes:
                            v = expand_node(node,c)
                            self.vectors[v] = word2vec[v]
                else:
                    for word in G.nodes():
                        self.vectors[word] = word2vec[word]

        del word2vec


    def get_node2vec(self):
        return self.vectors

# the nodes and their reduced dimensionality (using T-SNE) after retrieving their embeddings representation
# for each network size (small,medium,large)
#     def get_subgraphs_nodes(self,community_dict,n_components = 2):
#         community_nodes = tuple(sorted(community_dict.values(),key=lambda t: len(t),reverse=True))
#         size_dict = {}
#         vectors = self.vectors
#
#         # size_dict['large'] = [ str(u) for u in community_nodes[0]]
#         # size_dict['large_vec'] = TSNE(n_components=n_components).fit_transform([vectors[u] for u in size_dict['large']])
#
#         size_dict['small'] = [ [str(u) for u in nodes] for nodes in community_nodes[-2:]]
#         size_dict['small_vec'] = [TSNE(n_components=n_components).fit_transform([vectors[u] for u in nodes])
#                                                                                for nodes in size_dict['small']]
#
#         median = int(len(community_nodes) / 2)
#         size_dict['medium'] =  [ [str(u) for u in nodes] for nodes in community_nodes[median-1:median+1]]
#         size_dict['medium_vec'] =[TSNE(n_components=n_components).fit_transform([vectors[u] for u in nodes])
#                                                                                for nodes in size_dict['medium']]
#         return size_dict


    # def save_plot(self,community_dict,Algs = {'Kmeans':KMeans,'Birch':Birch},n_clusters = 3):
    #     import os
    #     os.environ["PATH"] += os.pathsep +\
    #                           'C://Users//galkampel.DESKTOP-KTJUKIM//Anaconda3//envs//thesis_env//Lib//site-packages//graphviz-2.38//release//bin'
    #
    #     G = self.graph.G
    #     size_dict = self.get_subgraphs_nodes(community_dict)
    #     sizes = ['small','medium'] #'large','medium',
    #     for name,Alg in Algs.items():
    #         for size in sizes:
    #             for i in range(len(size_dict[size])):
    #                 alg = Alg(n_clusters = n_clusters )
    #                 labels = alg.fit_predict(size_dict[size+'_vec'][i])
    #
    #
    #                 G_sub = G.subgraph(size_dict[size][i])
    #                 nx.draw(G_sub, pos=graphviz_layout(G_sub),arrows=False, with_labels=False, #,prog='dot'
    #                         node_list = size_dict[size][i] ,node_size=25, node_color=labels, cmap=plt.cm.Blues, alpha=0.8)
    #                 plt.title('A {} community size plot with {} (N = {})'.format(size,name,len(G_sub.nodes())))
    #                 plt.savefig('plots/{}{}_{}_p={}_q={}_d={}_r={}_l={}_k={}.svg'.
    #                             format(size,i+1,name,self.p,self.q,self.d,self.r,self.l,self.k),bbox_inches='tight')

    def save_embeddings(self, filename):
        fout = open(filename, 'w')
        node_num = len(self.vectors.keys())
        fout.write("{} {}\n".format(node_num, self.size))
        for node, vec in self.vectors.items():
            fout.write("{} {}\n".format(node,' '.join([str(x) for x in vec])))
        fout.close()





class EpochLogger(CallbackAny2Vec):
    def __init__(self,num_epochs,lr,params,is_community_level):
        self.num_epochs = num_epochs
        self.lr = lr
        self.params = params
        self.losses = []
        self.is_community_level = is_community_level

    def on_epoch_end(self, model):
        self.losses.append(model.get_latest_training_loss())
        if len(self.losses) == self.num_epochs :
            self.plot_lr()

    def plot_lr(self):
        import matplotlib.pyplot as plt
        lr = self.lr
        x = list(range(1,self.num_epochs+1))
        params_str = "_".join("{}={}".format(key, val) for key, val in self.params.items() if key != 'workers' )
        plt.plot(x, self.losses, color='b')
        plt.title("Node2vec loss with\n{},lr={}".format(params_str, lr))
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.savefig('plots_gensim/{}_lr={}.png'.format(params_str, lr))
        plt.close()

