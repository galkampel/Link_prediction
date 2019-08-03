### create graph as a benchmark

# import matplotlib.pyplot as plt
import numpy as np
from time import time
import networkx as nx
# from file2obj import save_labels,convert_file2obj,save_str
from itertools import chain
from igraph import *
from csv import reader
from collections import Counter,OrderedDict
import random
from copy import deepcopy
from networkx.algorithms.community.community_generators import LFR_benchmark_graph
# from joblib import Parallel,delayed
# from multiprocessing import cpu_count,Manager



#check 3 type of connection (high-high,high-low,low-low) for each
def check_node2vec_edges(G,community_dict,between_dict,k,num_examples=2000):

    importance_dict = {}
    for c,nodes in community_dict.items():
        G_sub = G.subgraph(nodes)
        ec = nx.eigenvector_centrality_numpy(G_sub)
        ec_ordered = OrderedDict(sorted(ec.items(),key = lambda k_v: k_v[1],reverse=True))
        ec_ordered = dict((k,v)  for k,v in ec_ordered.items() if v > 0)

def get_graph_stats(G,community_dict = None,count_communities = True):
    num_nodes = len(G.nodes())
    num_edges = len(G.edges())
    meadian_degree = np.median(list(zip(*G.degree()))[1])
    average_degree =  np.mean(list(zip(*G.degree()))[1])
    if count_communities:
        # diameter = Graph(edges=list(G.edges()), directed=False).diameter(directed=False)
        num_communities = len(community_dict)
        return num_nodes,num_edges,average_degree,meadian_degree,num_communities#,diameter

    rounded_degree = np.rint(average_degree)
    return num_nodes, num_edges, average_degree, rounded_degree, meadian_degree

class CommunityNetwork(object):
    def __init__(self,min_num_edges):
        #
        self.log = ''
        self.community_dict = {}
        self.between_edges_dict = {}
        self.communities_n2v_dict = None
        self.G = None
        self.min_num_edges = min_num_edges
        self.min_links = 50
        # self.num_cores = cpu_count()

    def get_G(self):
        return self.G
    def get_log(self):
        return self.log

    def get_community_dict(self):
        return self.community_dict

    def get_communities_n2v_dict(self):
        return self.communities_n2v_dict


    def get_between_edges_dict(self):
        return self.between_edges_dict

#N - #nodes, alpha- node_degree by power-law dist. (with exponent alpha)
#beta-community size by power-law dist. (with exponent beta)
#mu-fraction of links that are between-communities links (as mu goes larger it is harder to identify a community
#k - average degree
    def lfr_stats(self,params):
        log = ''
        log += 'LFR parameters:\n'
        log += 'N = {}\nalpha = {}\nbeta = {}\nmu = {}\nk_min = {}\nk_max = {}\nc_min = {}\n' \
               'c_max = {}\nseed = {}\n'\
            .format(params['N'],params['alpha'],params['beta'],params['mu'],params['k_min'],params['k_max'],
                    params['c_min'],params['c_max'],params['seed'])
        self.log += log


    def check_ex_communities(self,updated_dict,label_name):
        c_s = set(updated_dict.keys())
        ex_cs = []
        G = self.G
        for c, nodes in updated_dict.items():
            rest_cs = c_s.copy()
            rest_cs.remove(c)
            for c2 in rest_cs:
                if len(set(nodes).difference(set(updated_dict[c2]))) == 0:
                    for u in nodes:
                        G.nodes[u][label_name].remove(c)
                    print('community {} was removed'.format(c, len(set(nodes))))
                    ex_cs.append(c)

        if len(ex_cs) > 0:
            print('updated community dict size before community removal = {}'.format(len(updated_dict)))
            for c in ex_cs:
                if c in updated_dict:
                    del updated_dict[c]
            print('{} communities were removed'.format(len(ex_cs)))
            print('updated community dict size after community removal = {}\n'.format(len(updated_dict)))
        else:
            print("We're all good!\n")

    def check_community_dict(self,community_dict,label_name):
        G = self.G
        count_small_Cs = 0
        updated_dict = {}
        one_val_dict = {label:nodes for label,nodes in community_dict.items() if len(G.subgraph(nodes).edges()) == 0 }
        rest_val_dict = {label:nodes for label,nodes in community_dict.items() if len(G.subgraph(nodes).edges()) > 0}
        for C,nodes in rest_val_dict.items():
            if not nx.is_connected(G.subgraph(nodes)):
                print('Community {} with {} components\n'.format(C,nx.number_connected_components(G.subgraph(nodes))))
        # print(''.join('Is community {} connected? {}\n'.format(C,nx.is_connected(G.subgraph(nodes)))) for C,nodes in rest_val_dict.items())
        ### handle communities with only one node
        if len(one_val_dict.keys()) > 0:
            for label,nodes in one_val_dict.items():

                if len(nodes) > 1:
                    rest_val_dict[label] = nodes
                    continue
                counter = {}
                u = nodes[0]
                for v in G.neighbors(u):
                        for label2 in G.nodes[v][label_name]:
                            counter[label2] = counter.get(label2,0) + 1
                max_count = 0
                c = None
                for new_label,count in counter.items():
                    if count > max_count:
                        max_count = count
                        c = new_label
                # print('labels before removal: {}'.format(G.nodes[u][label_name]))
                G.nodes[u][label_name].remove(label)
                # G.nodes[u][label_name] = list()
                if c not in G.nodes[u][label_name]:
                    G.nodes[u][label_name].append(c)
                    updated_dict.setdefault(c,list(community_dict[c])).append(u)
                    if c in rest_val_dict:
                        rest_val_dict[c].append(u)
                    else:
                        one_val_dict[c].append(u)
                # print('labels after removal: {}'.format(G.nodes[u][label_name]))
                count_small_Cs += 1

        count_small_Cs = 0
        for label,nodes in rest_val_dict.items():
            # num_links = len(G.subgraph(nodes).edges())
            # if num_links <= self.min_links:
            if not nx.is_connected(G.subgraph(nodes)) or (len(nodes) - 1 == len(G.subgraph(nodes).edges())):

                count_small_Cs += 1
                counter = {}
                all_links = set(G.edges(nodes))
                within_links = set(G.subgraph(nodes).edges())
                rel_links = all_links.difference(within_links)
                for link in rel_links:
                    u,v = link[0],link[1]
                    if label in G.nodes[u][label_name]:
                        for label2 in G.nodes[v][label_name]:
                            counter[label2] = counter.get(label2,0) + 1
                    else:
                        for label2 in G.nodes[u][label_name]:
                            counter[label2] = counter.get(label2,0) + 1
                max_count = 0
                c = None
                for new_label,count in counter.items():
                    if count > max_count and new_label != label:
                        max_count = count
                        c = new_label

                for u in nodes:
                    G.nodes[u][label_name].remove(label)
                    G.nodes[u][label_name].append(c)
                updated_dict.setdefault(c,list(community_dict[c])).extend(nodes)
                rest_val_dict[c].extend(nodes)
                if label in updated_dict:
                    # print('commmunity {} with nodes {} should be removed'.format(label,[(u,G.nodes[u][label_name]) for u in nodes]))
                    # print('nodes = {}\nupdated dict nodes = {}'.format(nodes,updated_dict[label]))
                    # print('num links: {}'.format(num_links))
                    del updated_dict[label]
            else:
                if label not in updated_dict:
                    updated_dict[label] = list(nodes)
        # updated_dict2 = {}
        # for u in G.nodes():
        #     for label in G.nodes[u][label_name]:
        #         updated_dict2.setdefault(label,list()).append(u)
        #
        # print('count small Cs from community of size > 1 :\n {}'.format(count_small_Cs))
        # print('There are {} total nodes'.format(np.sum(len(G.nodes[u][label_name]) for u in G.nodes())))
        # print('community dict has {} communities and {} nodes'.format(len(community_dict.keys()),np.sum(len(nodes) for nodes in community_dict.values())))
        # check_ex_communities: removes communities that are completely in other community
        # self.check_ex_communities(updated_dict, label_name)
        print('updated dict has {} communities and {} nodes'.format(len(updated_dict.keys()),np.sum(len(nodes) for nodes in updated_dict.values())))

        #check the community is not in other community



        # for C,nodes in updated_dict.items():
        #     if not nx.is_connected(G.subgraph(nodes)):
        #         print('C = {} with {} components\nodes ={}'.format(C,nx.number_connected_components(G.subgraph(nodes)),nodes))
        # print('updated dict 2 has {} communities and {} nodes'.format(len(updated_dict2.keys()),np.sum(len(nodes) for nodes in updated_dict2.values())))

        # bad1 = [key for key,nodes in updated_dict.items() if len(G.subgraph(nodes).edges()) <= self.min_links]
        # print('bad nodes updated dict:\n{}'.format(bad1))
        # print('num fucks = {}'.format(np.sum(b in one_val_dict for b in bad1)))
        # print('bad nodes updated 2 dict:\n{}'.format([key for key,nodes in updated_dict.items() if len(G.subgraph(nodes).edges()) <= self.min_links]))
        #
        # print('missing link:\n{}'.format([(nodes,updated_dict2[key]) for key,nodes in updated_dict.items() if len(set(nodes).difference(set(updated_dict2[key]))) > 0]))
        # d = {key:val for key,val in updated_dict2.items() if key not in updated_dict}
        # if len(d.keys()) > 0:
        #     print("{}".format(d))
        #     print("there are {} problematic keys".format(np.sum([c in rest_val_dict for c in d.keys()])))
        #     print('Failure in dictionary creation')
        #     exit()

        return updated_dict


    # 0:None, 1:'multilevel',2:'infomap',3:'label_propagation'
    def create_lfr(self,has_communities,N,alpha,beta,mu,k_min,k_max,c_min,c_max,seed,model_type = 0): #maybe add max_iters
        G = LFR_benchmark_graph(n=N,tau1=alpha,tau2=beta,mu=mu,min_degree=k_min,max_degree=k_max,min_community=c_min,
                                max_community=c_max,seed=seed)
        G.remove_edges_from(list(filter(lambda edge:  edge[0] == edge[1],G.edges())))
        self.G = G
        # print('self loops:\n{}'.format(list(filter(lambda edge:  edge[0] == edge[1],G.edges()))))
        communities_n2v_dict = None
        communities = {frozenset(G.nodes[v]['community']) for v in G}
        community_dict = {i:list(nodes) for i,nodes in enumerate(communities)}
        community_dict = self.check_connected_communities(community_dict)
        # reseting community label
        for v in G.nodes():
            G.nodes[v]['community'] = list()
        if not has_communities:
            model = None
            ig_G = Graph( edges=list(G.edges()), directed=False)
            if model_type == 1:
                model = ig_G.community_multilevel(return_levels=False)
                print("CD using multilevel")
            elif model_type == 2:
                model = ig_G.community_infomap()
                print("CD using infomap")
            else: #label propagation
                print("CD using label propagation")
                model = ig_G.community_label_propagation()
            communities_n2v = model.as_cover()._clusters
            communities_n2v_dict = {i:list(nodes) for i,nodes in enumerate(communities_n2v)}
            for c in communities_n2v_dict.keys():
                for u in communities_n2v_dict[c]:
                    G.nodes[u].setdefault('community', []).append(c)
            print('community n2v dict 2:\n{}'.format(communities_n2v_dict))
            communities_n2v_dict = self.check_community_dict(communities_n2v_dict,'community')

        for c in community_dict.keys():
            for u in community_dict[c]:
                G.nodes[u].setdefault('label', []).append(c)

        # print('num non-connected components = {}'.format(len(community_dict.keys()) - np.sum([nx.is_connected(G.subgraph(community)) for community in community_dict.values()]) ))
        # for i,component in enumerate(nx.connected_components(G)):
        #     print('component_{} has {} nodes and {} edges'.format(i,len(component),len(G.subgraph(component).edges())))
        # community_dict = self.check_community_dict(community_dict,'label')
        # print('num non-connected components = {}'.format(len(community_dict.keys()) - np.sum([nx.is_connected(G.subgraph(community)) for community in community_dict.values()]) ))
        self.community_dict = self.check_community_dict(community_dict,'label')
        self.communities_n2v_dict = communities_n2v_dict
        for c,nodes in self.community_dict.items():

            diameter_vec = Graph( edges=list(G.subgraph(nodes).edges()), directed=False).get_diameter(directed = False)
            max_num_it = 10 if len(nodes) < 1000 else 300 if len(diameter_vec) - 1 == 4 else 150
            if len(diameter_vec) - 1 > 2:
                num_it = 0
                while len(diameter_vec) - 1 > 2 and num_it < max_num_it:
                    new_edge = (diameter_vec[0],diameter_vec[-1])
                    G.add_edge(*new_edge)
                    diameter_vec = Graph( edges=list(G.subgraph(nodes).edges()), directed=False).get_diameter(directed = False)
                    num_it += 1
                print('done')

            # diameter = Graph( edges=list(G.subgraph(nodes).edges()), directed=False).diameter(directed = False)
            print('community {} with diameter of length:{}'.format(c,len(diameter_vec)-1))


        N,E,k_avg,k_median,C,diameter = [None] * 6
        if communities_n2v_dict is not None:
            N,E,k_avg,k_median,C,diameter = get_graph_stats(G,communities_n2v_dict)
        else:
            N,E,k_avg,k_median,C,diameter = get_graph_stats(G,self.community_dict)
            # if diameter <= 2:
            #     print('community {} has {} nodes and a diameter of size {}'.format(key,len(nodes),diameter))
            # mst = list(nx.minimum_spanning_edges(G.subgraph(nodes),data=False))
            # print('community {} has diameter of size {}'.format(key,mst))

        log = 'Graph parameters:\n'
        log += 'N = {0}\nE = {1}\nC = {2}\nk_avg = {3:.3f}\nk_meadian = {4:.3f}\ndiameter = {5}\n'.format(N,E,C,k_avg,k_median,diameter)
        self.log += log
        # counter = Counter(len(community) for community in communities.values())
        # plt.hist(counter.keys(),range=(50,500),bins=11)
        # plt.title('Communities size distribution')
        # plt.xlabel('community size')
        # plt.show()

        # communities_size = np.unique([len(community) for community in communities.values()])
        # communities_links_size = np.unique([len(G.subgraph(community).edges()) for community in communities.values()])
        # print('communities sizes = {}\nnumber of links per communities = {}'.format(communities_size,communities_links_size))

        if C - np.sum([nx.is_connected(G.subgraph(community)) for community in community_dict.values()]) < 0:
            print('total components: {}'.format(np.sum([nx.number_connected_components(self.G.subgraph(community)) for community in community_dict.values()])))
            print('Fail to create connected communities.\nStop execution')
            exit()
        else:
            # for i,nodes in enumerate(community_dict.values()):
            #     print('community {} has {} nodes and {} edges'.format(i,len(nodes),len(self.G.subgraph(nodes).edges())))
            #     if len(nodes) == 1:
            #         print('one node subgraph\nnode = {}\nedges = {}'.format(nodes,G.subgraph(nodes).edges()))
            print('All the communities are connected')


    def get_min_labels(self,labels,community_length):
        i_min = -1
        C_min = np.inf
        for i,label in enumerate(labels):
            C = community_length[label]
            if C < C_min:
                C_min = C
                i_min = i
        return labels[i_min]


    def parallel_between_edges_dict(self,i,u,v,between_edges_dict,community_dict):
        G = self.G
        labels1,labels2 = G.nodes[u]['label'],G.nodes[v]['label']
        label1 = self.get_min_labels(labels1,community_dict)
        label2 = self.get_min_labels(labels2,community_dict)
        print('iteration number: {}'.format(i))
        print('label1 = {}\nlabel2 = {}'.format(label1,label2))
        if label1 < label2:
            tmp = {(label1,label2):(u,v)}
            between_edges_dict.update(tmp)
        else:
            tmp = {(label2,label1):(v,u)}
            between_edges_dict.update(tmp)

    def create_between_edges_dict(self,verbose=False):
        between_edges_dict = self.between_edges_dict
        community_dict = self.community_dict
        G = self.G
        G_rest = G.copy()
        for nodes in community_dict.values():
            G_rest.remove_edges_from(G.subgraph(nodes).edges())

        log = 'Additional information:\n'
        log += 'there are {} between edges\n'.format(len(G_rest.edges()))
        #runnig over all between edges

        # manager = Manager()
        # between_edges_dict = manager.dict()
        # start = time()
        # Parallel(n_jobs=self.num_cores)(delayed(self.parallel_between_edges_dict)(i,u,v,between_edges_dict,community_dict)
        #                            for i,(u,v) in enumerate(G_rest.edges()))

        # between_edges_dict = dict(between_edges_dict)
        community_length = {label:len(G.subgraph(community_dict[label]).edges()) for label,nodes in community_dict.items()}
        for (u,v) in G_rest.edges():
            labels1,labels2 = G.nodes[u]['label'],G.nodes[v]['label']
            label1 = self.get_min_labels(labels1,community_length)
            label2 = self.get_min_labels(labels2,community_length)
            if label1 < label2:
                between_edges_dict.setdefault((label1,label2),[]).append((u,v))
            else:
                between_edges_dict.setdefault((label2,label1),[]).append((v,u))
            # for label1 in labels1:
            #     for label2 in labels2:
            #         if label1 < label2:
            #             between_edges_dict.setdefault((label1,label2),[]).append((u,v))
            #         else:
            #             between_edges_dict.setdefault((label2,label1),[]).append((v,u))

        #if too little information (less than min_num_edges), than it is problematic to make a prediction between those 2 communities
        min_num_edges = self.min_num_edges
        between_edges_dict = {key:edges  for key,edges in between_edges_dict.items() if len(edges)>= min_num_edges}

        #### ADDITION ##############
        # for key,edges in between_edges_dict.items():
        #
        #     sorted(nx.connected_components(G.edge_subgraph(edges)),key=len,reverse = True)
        #     print('key {} has {} components'.format(key,nx.number_connected_components(G.edge_subgraph(edges))))
        #### END ADDITION ##############

        self.between_edges_dict = between_edges_dict
        log += 'there are {} within edges keys with total of {} within edges\n'.\
            format(len(community_dict.keys()),np.sum(len(G.subgraph(nodes).edges()) for nodes in community_dict.values()))
        log += 'there are {} between edges keys with total of {} between edges\n'.\
            format(len(between_edges_dict.keys()),np.sum(len(edges) for edges in between_edges_dict.values()))
        # num_edges_per_community = np.unique([len(edges) for edges in  between_edges_dict.values()])
        # log += 'num_edges per between-community = {}\n'.format(num_edges_per_community)
        self.log += log



    def check_connected_communities(self,community_dict):
        count = 0
        new_community_dict = {}
        for nodes in community_dict.values():
            # print('there are {} nodes and {} components'.format(len(nodes),nx.number_connected_components(self.G.subgraph(nodes))))
            for component in nx.connected_components(self.G.subgraph(nodes)):
                # print('component:\n{}'.format(component))
                new_community_dict[str(count)] = list(component)
                count += 1
        return new_community_dict


    def get_real_data(self,edgelist_file,groups_file,has_communities,model_type):
        community_dict = {}
        if  'Flickr' in edgelist_file or 'BlogCatalog' in edgelist_file:
            G = nx.read_edgelist(edgelist_file,delimiter=",")
            self.G = nx.relabel_nodes(G, lambda x: int(x))
            print("there are {} nodes and {} edges".format(len(self.G.nodes()),len(self.G.edges())))
            file_reader = reader(open(groups_file, newline=''), delimiter=',')#, quotechar='|')
            for row in file_reader:
                val,key = row
                community_dict.setdefault(key,[]).append(int(val))

            # community_dict = self.check_connected_communities(community_dict)
            # print('community dict has {} keys and {} values'.format(len(community_dict.keys()), np.sum(
            #     len(val) for val in community_dict.values())))
        #if have more data
        # else:

        # communities_n2v_dict = None
        # if not has_communities:
        #     model = None
        #     ig_G = Graph( edges=list(self.G.edges()), directed=False)
        #     if model_type == 1:
        #         model = ig_G.community_multilevel(return_levels=False)
        #         print("CD using multilevel")
        #     elif model_type == 2:
        #         model = ig_G.community_infomap()
        #         print("CD using infomap")
        #     else: #label propagation
        #         print("CD using label propagation")
        #         model = ig_G.community_label_propagation()
        #     communities_n2v = model.as_cover()._clusters
        #     communities_n2v_dict = {i:list(nodes) for i,nodes in enumerate(communities_n2v)}
        #
        #     for c in communities_n2v_dict.keys():
        #         for u in communities_n2v_dict[c]:
        #             self.G.nodes[u].setdefault('community', []).append(c)
        #     communities_n2v_dict = self.check_community_dict(communities_n2v_dict,'community')
        # for c in community_dict.keys():
        #     for u in community_dict[c]:
        #         self.G.nodes[u].setdefault('label', []).append(c)
        # # print('label dict:\n{}'.format(nx.get_node_attributes(G,'label')))
        # # community_dict = self.check_community_dict(community_dict,'label')

        # self.community_dict = self.check_community_dict(community_dict,'label')
        # # print('community dict has {} keys and {} values'.format(len(self.community_dict.keys()), np.sum(
        # #     len(val) for val in self.community_dict.values())))
        # self.communities_n2v_dict = communities_n2v_dict

        # N,E,k_avg,k_median,C = [None] * 5
        # start = time()
        # if communities_n2v_dict is not None:
        #     N,E,k_avg,k_median,C,diameter = get_graph_stats(self.G,communities_n2v_dict)
        # else:
        #     N,E,k_avg,k_median,C,diameter = get_graph_stats(self.G,self.community_dict)
        for c in community_dict.keys():
            for u in community_dict[c]:
                self.G.nodes[u].setdefault('label', []).append(c)
        avg_Cs = 0
        for u in self.G.nodes():
            avg_Cs += len(self.G.nodes[u]['label'])
        N,E,k_avg,k_median,C = get_graph_stats(self.G,self.community_dict)
        avg_Cs /= N
        log = 'Graph parameters:\n'
        log += 'N = {0}\nE = {1}\nC = {2}\nk_avg = {3:.3f}\nk_meadian = {4:.3f}\nC_avg = {5:3f}\n'.format(N,E,C,k_avg,k_median,avg_Cs)
        self.log += log
        self.community_dict = community_dict
        # if C - np.sum([nx.is_connected(self.G.subgraph(community)) for community in self.community_dict.values()]) > 0:
        #     print('total components: {}'.format(np.sum([nx.number_connected_components(self.G.subgraph(community)) for community in self.community_dict.values()])))
        #     for key,nodes in community_dict.items():
        #         print('community number {} has {} components'.format(key,nx.number_connected_components(self.G.subgraph(nodes))) )
        #     print('G is connected?\n{}'.format(nx.is_connected(self.G)))
        #     print('num componenets1 = {}'.format(np.sum([nx.is_connected(self.G.subgraph(community)) for community in self.community_dict.values()])))
        #     print('Fail to create connected communities.\nStop execution')
        #     exit()
        # else:
            # for i,nodes in enumerate(community_dict.values()):
            #     print('community {} has {} nodes and {} edges'.format(i,len(nodes),len(self.G.subgraph(nodes).edges())))
            # print('All the communities are connected')






# if __name__ == '__main__':
#     ##large network tau1=3,tau2=1.5,mu=0.3,k_min=20,k_max=100,c_min=20,c_max=500,seed={0,1}
#     params = convert_file2obj('input_community.json')
#     communiy_graph = CommunityNetwork(min_num_edges=10)
#     communiy_graph.get_real_data('datasets\\BlogCatalog-dataset\\edges.csv',None)
#     communiy_graph.lfr_stats(params)
#     # communiy_graph.get_lfr(100000,2,1.05,0.3,20,100,50,500,1)
#     communiy_graph.create_lfr(params["N"],params["alpha"],params["beta"],params["mu"],params["k_min"],params["k_max"],
#                            params["c_min"],params["c_max"],params["seed"])
#     communiy_graph.get_between_edges_dict(verbose=True)
#
#     save_str(communiy_graph.get_log(),'LFR_graphs/LFR{}_stats.txt'.format(params["id"]))
