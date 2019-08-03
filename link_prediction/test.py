
import networkx as nx
from networkx.algorithms.community.community_generators import LFR_benchmark_graph
import numpy as np
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout
# from matplotlib.transforms import Bbox
from time import time
import random
from igraph import *
import igraph as ig
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import os
fig_size = plt.rcParams["figure.figsize"]
print(fig_size)
fig_size[0] = 9.6#8
fig_size[1] = 7.6#6
# os.environ["PATH"] += os.pathsep +\
#                       'C://Users//galkampel.DESKTOP-KTJUKIM//Anaconda3//envs//thesis_env//Lib//site-packages//graphviz-2.38//release//bin'
def main():
    random.seed(1)
    G = Graph.SBM(n=10000, pref_matrix=[[0.05, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005],
                                        [0.005, 0.05, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005],
                                        [0.005, 0.005, 0.05, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005],
                                        [0.005, 0.005, 0.005, 0.05, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005],
                                        [0.005, 0.005, 0.005, 0.005, 0.05, 0.005, 0.005, 0.005, 0.005, 0.005],
                                        [0.005, 0.005, 0.005, 0.005, 0.005, 0.05, 0.005, 0.005, 0.005, 0.005],
                                        [0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.05, 0.005, 0.005, 0.005],
                                        [0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.05, 0.005, 0.005],
                                        [0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.05, 0.005],
                                        [0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.05]],
                  block_sizes = [1500, 500, 250, 1250, 2000, 2500, 2000, 1500, 750, 1250])
    #[1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]


    print(len(G.get_edgelist()))
    print('Done creating a graph')
    # community_infomap
    # community_label_propagation
    # community_multilevel
    CD = G.community_multilevel()
    print('Done CD')
    labels = np.array(CD._membership)
    for i in range(10):
        print('There are {} of label {}'.format(np.sum(np.array(labels) == i),i))
    print(len(labels))
    print("there are {} communities".format(len((np.unique(labels)))))
    G_2 = nx.from_edgelist(G.get_edgelist())
    G_2
    exit()
    n = 300
    seed = 1
    np.random.seed(seed)


    G = LFR_benchmark_graph(n,2,1.05,0.15,min_degree=5,max_degree=100,min_community=20,max_community=110,seed=seed)
    g = Graph( edges=list(G.edges()), directed=False) #vertex_attrs={"label":vertices},
    egn_vec = np.array(g.evcent())

    print("max val = {}\nmean val = {}\n min val = {}".format(egn_vec.max(),egn_vec.mean(),egn_vec.min()))
    print(egn_vec[egn_vec > 0.71].argsort())
    # print([(edge.source,edge.target) for edge in g.es])
    start = time()
    # pos = nx.spring_layout(G)
    pos = graphviz_layout(G)
    nx.draw(G,pos=pos,node_size=25)
    # plt.show()
    # plt.savefig('plot3.png',format='png')
    plt.title("example for something")
    plt.savefig('plot2.svg',format='svg',bbox_inches='tight')
    # plt.savefig('plot3.svg',format='svg',dpi = 300)
    # plt.savefig('plot4.svg',format='svg',dpi = 600)
    # plt.savefig('plot5.svg',format='svg',dpi = 1200)
    # plt.savefig('plot6.eps',format='eps',dpi = 1000)
    # plt.subplot()
    n_rows = 1
    n_cols = 1
    # num_subplot = n_rows * 100 + n_cols * 10
    # for i in range(n_rows):
    #     for j in range(n_cols):
    #         labels = np.random.choice(3, n)
    #         # labels = ['123' if label == 0 else '25' if label == 1 else '76' for label in labels]
    #         plt.subplot(num_subplot + (i * n_cols + j + 1))
    #         nx.draw(G,pos=pos,node_size=25,node_color = labels,cmap = plt.cm.Blues,alpha=0.8)
    #         plt.title('A {} community size plot'.format('medium'))


    # nx.draw_networkx_nodes(G,pos= pos,alpha=0.8,node_size = 50,node_color = labels)
    # nx.draw_networkx_edges(G,pos=pos,edge_color='grey')
    # nx.draw_networkx(G, pos=draw_spring(G), arrows=False, with_labels=False, alpha=0.8, node_size=50,
    #                  node_color=labels)
    # print("it takes {:3f} seconds to draw a graph".format(time() - start))
    # path = 'plots/'
    # plt.savefig(path+'my_first_plot=3.png',bbox_inches='tight')











if __name__ == '__main__':
    main()

