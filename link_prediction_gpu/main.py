
import networkx as nx
from graph import Graph
from preprocess import Preprocess
from node2vec import Node2vec
from walker import expand_node
from model import Model
from edge2vec import *
from community import CommunityNetwork,get_graph_stats
from file2obj import save_to_gzip,read_gzip_object,convert_file2obj,save_to_csv,save_str,save_res_to_csv,read_csv_params
from time import time
import igraph as ig
from itertools import chain,product
from multiprocessing import cpu_count
import matplotlib.pyplot as plt
import pandas as pd
import gc
gc.collect()

def create_dateset(data_name,files_name,CD_model,has_communities=True,is_synthetic=False):
    #'input_community.json'
    if is_synthetic:
        params = convert_file2obj('input_syn_community.json')
        community_graph = CommunityNetwork(min_num_edges=8)
        community_graph.lfr_stats(params)
        community_graph.create_lfr(has_communities,params["N"],params["alpha"],params["beta"],params["mu"],params["k_min"],
                                   params["k_max"], params["c_min"],params["c_max"],params["seed"],CD_model)

        community_graph.create_between_edges_dict(verbose=True)
        save_str(community_graph.get_log(),'graphs/{}_{}_stats.txt'.format(data_name,CD_model))
        save_to_gzip(community_graph.get_G(),'{}_{}_{}.gz'.format(data_name,CD_model,files_name[0]))
        save_to_gzip(community_graph.get_between_edges_dict(),'{}_{}_{}.gz'.format(data_name,CD_model,files_name[2]))

        community_dict = community_graph.get_community_dict()
        if has_communities:
            save_to_gzip(community_dict,'{}_{}_{}.gz'.format(data_name,CD_model,files_name[1]))
        else:
            print("new community dict was saved")
            save_to_gzip(community_graph.get_communities_n2v_dict(),'{}_{}_{}.gz'.format(data_name,CD_model,files_name[1]))
            print('new community dict:\n'.format(community_graph.get_communities_n2v_dict()))

        return community_dict

    #real data
    else:
        params = convert_file2obj('input_real_community.json')
        community_graph = CommunityNetwork(min_num_edges=6)
        community_graph.get_real_data(params['edgelist_file'],params['groups_file'],has_communities,CD_model)
        community_graph.create_between_edges_dict()
        save_str(community_graph.get_log(),'graphs/{}_{}_stats.txt'.format(data_name,CD_model))
        save_to_gzip(community_graph.get_G(),'{}_{}_{}.gz'.format(data_name,CD_model,files_name[0]))
        save_to_gzip(community_graph.get_between_edges_dict(),'{}_{}_{}.gz'.format(data_name,CD_model,files_name[2]))

        community_dict = community_graph.get_community_dict()
        print('community dict has {} keys and {} values'.format(len(community_dict.keys()), np.sum(
            len(val) for val in community_dict.values())))
        if has_communities:
            save_to_gzip(community_dict,'{}_{}_{}.gz'.format(data_name,CD_model,files_name[1]))
        else:
            print("new community dict was saved")
            save_to_gzip(community_graph.get_communities_n2v_dict(),'{}_{}_{}.gz'.format(data_name,CD_model,files_name[1]))
            print('new community dict:\n'.format(community_graph.get_communities_n2v_dict()))

        return community_dict


#create 'a given' graph to predict from
def preprocess_G(data_name,files_name,CD_model,has_communities,community_dict,seed = 1,frac_within=0.5,frac_between=0.5):
    G = read_gzip_object('{}_{}_{}.gz'.format(data_name,CD_model,files_name[0]))
    # community_dict = read_gzip_object('{}_{}_{}.gz'.format(data_name,CD_model,files_name[1]))
    between_edges_dict = read_gzip_object('{}_{}_{}.gz'.format(data_name,CD_model,files_name[2]))
    print('Start preprocessing...')
    #default seed = 1
    preprocess = Preprocess(G,community_dict,between_edges_dict,frac_within,frac_between,seed = seed)
    start = time()
    preprocess.set_new_G('{}_{}_{}_{}_{}'.format(data_name,CD_model,int(has_communities),seed,files_name[4]))
    print('It took {:.3f} seconds to create new G and test set'.format(time()-start))
    start = time()
    preprocess.save_train_obj('{}_{}_{}_{}_{}'.format(data_name,CD_model,int(has_communities),seed,files_name[3]))
    print('It took {:.3f} seconds to create new training set'.format(time()-start))
    save_to_gzip(preprocess.get_G(),'{}_{}_{}_{}_{}.gz'.format(data_name,CD_model,int(has_communities),seed,files_name[0]))


def partition_community_dict(community_dict,G,data_name,CD_model,seed,thres=10000,to_save=True):

    df = pd.DataFrame.from_dict({key:get_graph_stats(G.subgraph(nodes),count_communities=False) for key,nodes in community_dict.items()},
                                    orient='index',columns = ['nodes','edges','avg degree','r_avg degree','med degree'])
    is_outlier = (((df['r_avg degree'] - df['r_avg degree'].mean()).abs() > 2 * df['r_avg degree'].std())
                    & ((df['r_avg degree'] - df['r_avg degree'].mean()).abs() > 2 * df['r_avg degree'].std()))
    c_outlies = df[is_outlier].index.tolist()
    rel_df = df[~is_outlier]
    k_rounded_meds = tuple(rel_df.groupby(['r_avg degree', 'med degree']).groups.keys())
    c_to_parj = {}
    num_edges = 0
    N = len(k_rounded_meds)
    j = 0
    for i,k_rounded_med in enumerate(k_rounded_meds):
        k_rounded,k_med = k_rounded_med
        tmp_df = df[(df['r_avg degree'] == k_rounded) & (df['med degree'] == k_med) ]
        for c in tmp_df.index.tolist():
            c_to_parj[c] = j

        num_edges += tmp_df['edges'].sum()
        if num_edges > thres or i == N-1:
            j += 1
            num_edges = 0

    for c in c_outlies:
        c_to_parj[c] = j
        j += 1
            # G_tmp = G.subgraph(list(chain(* dict_i.values())))
            # if nx.is_connected(G_tmp):
            #     ordered_communities.append(dict_i)
            #     partitions.append([dict_i])
            # else:
            #     list_i = []
            #     print('1) there are {} components'.format(nx.number_connected_components(G_tmp)))
            #     for G_sub in nx.connected_component_subgraphs(G_tmp):
            #         sub_nodes = set(G_sub.nodes())
            #         print('G sub nodes\n{}'.format(sub_nodes))
            #         list_i.append({c:dict_i[c] for c in dict_i.keys() if len(set(dict_i[c]).difference(sub_nodes)) == 0})
            #         print('2) there are {} communities\nThere are {} communities in component'.format(
            #             len(dict_i.keys()),len(list_i[-1].keys())))
            #         ordered_communities.append(list_i[-1])
            #     partitions.append(list_i)


    # for c in c_outlies:
    #     partitions.append({c: community_dict[c]})
    #     ordered_communities.append({c:community_dict[c]})

    if to_save:
        df.to_csv("df_{}_{}_{}.csv".format(data_name,CD_model,seed))
        save_to_gzip(c_to_parj,'community_to_partition_{}_{}_{}.gz'.format(data_name,CD_model,seed))

    return c_to_parj,j+1


def node2embeddings(params,data_name,files_name,CD_model,has_communities,community_dict,seed):
    # community_dict = read_gzip_object('{}_{}_{}.gz'.format(data_name,CD_model,files_name[1]))
    print('Node2vec phase (graph-wise)')
    graph = Graph()
    graph.read_edgelist('{}_{}_{}_{}_{}.gz'.format(data_name,CD_model,int(has_communities),seed,files_name[0]))
    start = time()
    node2vec = Node2vec(graph.G,params = params,community_dict=community_dict,size=params["d"],window=params["k"],
                        workers=cpu_count())
    print('It took {:.3f} seconds to create nodes representations'.format(time()-start))
    # save_to_gzip(node2vec.get_node2vec(),'{}_{}_{}_{}_{}_{}_{}.gz'.format(data_name,params_str,CD_model,int(has_communities),
    return node2vec.get_node2vec()

def node2embeddings_community(community_dict,c_to_partition,best_params_i,G,rho,r,d,k,to_expand):#,to_max = False): #is_community_level=False
    #create a community-level node2vec, and then merge all node2vecs to a single dictionary
    # graph = Graph()
    # graph.read_edgelist('{}_{}_{}_{}_{}.gz'.format(data_name,CD_model,int(has_communities),seed,files_name[0]))
    start = time()
    node2vec = Node2vec(G,best_params=best_params_i,community_dict=community_dict,c_to_partition = c_to_partition,
                        rho = rho,r=r,is_community_level = True, to_expand=to_expand, size=d,window=k,workers=cpu_count())
    print('It took {:.3f} seconds to create nodes representations'.format(time()-start))
    return node2vec.get_node2vec()


def edge2community_edge(data,is_within):
    if is_within:
        within_pos_edges = list((expand_node(link[0],c),expand_node(link[1],c) )
                                for c,links in data['pos']['within'].items() for link in links)
        within_neg_edges = list((expand_node(link[0],c),expand_node(link[1],c) )
                                for c,links in data['neg']['within'].items() for link in links)
        return within_pos_edges,within_neg_edges
    else:
        between_pos_edges = list((expand_node(link[0],c1),expand_node(link[1],c2) )
                                for (c1,c2),links in data['pos']['between'].items() for link in links)
        between_neg_edges = list((expand_node(link[0],c1),expand_node(link[1],c2) )
                                for (c1,c2),links in data['neg']['between'].items() for link in links)
        return between_pos_edges,between_neg_edges


def create_train_test(model,data_obj,bin_op,dim,is_community_level = False,to_expand = False):

    if is_community_level and to_expand:
        within_pos_edges,within_neg_edges = edge2community_edge(data_obj,is_within=True)
        between_pos_edges,between_neg_edges = edge2community_edge(data_obj,is_within=False)
        data_within_pos = model.edge2vec(within_pos_edges,bin_op,dim,True)
        data_within_neg = model.edge2vec(within_neg_edges,bin_op,dim,False)
        data_within = merge_edges(data_within_pos,data_within_neg)
        data_between_pos = model.edge2vec(between_pos_edges,bin_op,dim,True)
        data_between_neg = model.edge2vec(between_neg_edges,bin_op,dim,False)
        data_between = merge_edges(data_between_pos,data_between_neg)
        return data_within,data_between
    else:
        data_within_pos = model.edge2vec(list(chain(*data_obj['pos']['within'].values())),bin_op,dim,True)
        data_within_neg = model.edge2vec(list(chain(*data_obj['neg']['within'].values())),bin_op,dim,False)
        data_within = merge_edges(data_within_pos,data_within_neg)
        data_between_pos = model.edge2vec(list(chain(*data_obj['pos']['between'].values())),bin_op,dim,True)
        data_between_neg = model.edge2vec(list(chain(*data_obj['neg']['between'].values())),bin_op,dim,False)
        data_between = merge_edges(data_between_pos,data_between_neg)
        return data_within,data_between


#create 2 datasets: one for within edges, and the other for between edges
#using Hadamard because it was the best binary operator according to the article

def nodes2edge(vectors,files_name,CD_model,data_name,seed,has_communities,is_community_level,to_expand=False): #params_str

    train_obj = read_gzip_object('{}_{}_{}_{}_train.gz'.format(data_name,CD_model,int(has_communities),seed,files_name[3]))
    test_obj = read_gzip_object('{}_{}_{}_{}_{}.gz'.format(data_name,CD_model,int(has_communities),seed,files_name[4]))
    # vectors = read_gzip_object('{}_{}_{}_{}_{}_{}_{}.gz'.format(data_name,params_str,CD_model,int(has_communities),
    #                                                                        int(is_community_level),seed,files_name[-2]))
    print("start converting nodes to edges")
    d = list(vectors.values())[0].shape[0]
    model = Edge2vec(vectors)
    start = time()
    train_within,train_between = create_train_test(model,train_obj,Hadamard,d,is_community_level,to_expand)
    test_within,test_between = create_train_test(model,test_obj,Hadamard,d,is_community_level,to_expand)

    print('It takes {:.3f} seconds to create edge embeddings'.format(time() - start))

        # save 4 datasets (within/between for train/test)
    train = {'within': train_within,'between':train_between}
    test = {'within': test_within,'between':test_between}
    return train,test


def link_prediction(model_name,has_two_learners,is_community_level,train_dict,test_dict,model_params_path ,n2v_params,
                    output,only_validate=False,params_str=None):
    model_params = None
    if model_params_path is not None:
        model_params = convert_file2obj(model_params_path)
    model = Model(model_name,model_params,has_two_learners,is_community_level)
    if model_name == 'Logistic Regression':
        measures = model.get_measures(train_dict,test_dict,model_name,only_validate)
        output.append(n2v_params+measures)
    else:
        model.get_MLP_measures(train_dict,test_dict,output,n2v_params)

#has_communities:
# if 0 then need to use community detection algorithm to create communities
#CD_model:
# 0 if no need of CD algorithm, 1 for multilevel, 2 for infomap and 3 for label propagation


#is_community_level:
# 1 if community-level based node2vec should be created else (0) graph-level

# two types of prediction:
# type '1' - ignore community knowledge
# type '2' - differentiate between within and between community links
def exe(files_name,num_to_save,data_name,CD_model,model_name,has_two_learners,has_communities,is_community_level,
        seed,model_params_path=None,num_combinations = 40):

    all_params = convert_file2obj('all_params.json')
    # all_params = convert_file2obj('all_params_tmp.json')
    all_params["workers"] = cpu_count()
    output = []
    header = ["p","q","d","r","l","k","F1 within","AP within","Accuracy within",
              "F1 between","AP between","Accuracy between","F1 total","AP total","Accuracy total","params"]
    output.append(header)
    counter = 1
    ps, qs, d, r, ls, ks = list(all_params.values())[:6]
    all_combinations = [(p,q,d,r,l,k) for p in ps for q in qs for l in ls for k in ks  if l > k]
    # all_combinations = [(0.25, 0.25, 128, 16, 8, 3), (0.25, 0.5, 128, 16, 8, 2), (0.25, 1, 128, 16, 8, 2),
    #                     (0.25, 2, 128, 16, 8, 4), (1, 0.5, 128, 16, 5, 3), (1, 0.5, 128, 16, 5, 4),
    #                     (1,1,128,16,5,3),(2,0.25,128,16,5,3),(2,0.25,128,16,5,4),(2,0.5,128,16,5,3),
    #                     (2, 0.5, 128, 16, 5, 4),(2,1,128,16,5,3),(2,1,128,16,5,4),
    #                     (4,0.25,128,16,5,3),(4,0.25,128,16,5,4), (4, 0.5, 128, 16, 5, 3),
    #                     (4,0.5,128,16,5,4),(4,1,128,16,5,3),(4,1,128,16,5,4),
    #                     (4,2,128,16,5,3),(4, 4, 128, 16, 5, 3)
    #                     ]
    # np.random.seed(1)
    # all_combinations = list(product(all_params["p"],all_params["q"],all_params["d"],all_params["r"],
    #                                 all_params["l"],all_params["k"]))
    # all_combinations = [(0.5, 0.25, 128, 16, 5, 3),(0.5,0.25,128,16,5,4),(1,0.25,128,16,5,3),(1,0.25,128,16,5,4),(1,0.5,128,16,5,3),
    #                     (1, 0.5, 128, 16, 5, 4),(1,1,128,16,5,3),(2,0.25,128,16,5,3),(2,0.25,128,16,5,4),(2,0.5,128,16,5,3),
    #                     (2, 0.5, 128, 16, 5, 4),(2,1,128,16,5,3),(2,1,128,16,5,4),(4,0.25,128,16,5,3),(4,0.25,128,16,5,4),
    #                     (4, 0.5, 128, 16, 5, 3),(4,0.5,128,16,5,4),(4,1,128,16,5,3),(4,1,128,16,5,4),(4,2,128,16,5,3),
    #                     (4, 4, 128, 16, 5, 3)]
    # p_qs = [(all_params["p"][i], all_params["q"][j]) for i in range(len(all_params["p"])) for j in range(i, len(all_params["q"]))]
    # all_combinations = list(map(lambda el: el[0] + el[1], list(product(p_qs, list(product(all_params["d"], all_params["r"], all_params["l"],all_params["k"]))))))
    # idxes = np.random.choice(range(len(all_combinations)),num_combinations,replace = False)
    # cand_params = [all_combinations[idx] for idx in idxes]
    workers = all_params["workers"]
    # for cand_param in cand_params:
    check_gensim = False
    community_dict = read_gzip_object('{}_{}_{}.gz'.format(data_name, CD_model, files_name[1]))
    community_dict = {c:[str(u) for u in nodes] for c,nodes in community_dict.items()}
    for cand_param in all_combinations:
        p,q,d,r,l,k = cand_param
        params = {"p": p,"q": q, "d":d,"r":r,"l":l,"k":k,"workers":workers}
        n2v_params = list(params.values())[:-1]
        print("Node2vec parameters: {}".format(params))
        params_str = "_".join("{}={}".format(key,val) for key,val in params.items() if key !='workers')
        n2v_dict = node2embeddings(params,data_name,files_name,CD_model,has_communities,community_dict,seed)
        print('finished creating node2vec')
        if check_gensim:
            continue
        train_dict,test_dict = nodes2edge(n2v_dict,files_name,CD_model,data_name,seed,has_communities,is_community_level)
        link_prediction(model_name,has_two_learners,is_community_level,train_dict,test_dict,
                        model_params_path,n2v_params,output,params_str=params_str)
        if counter % num_to_save == 0:
            save_res_to_csv(int(counter / num_to_save),data_name,CD_model,has_communities,
                            is_community_level,has_two_learners,seed,model_name,output)
            output = []
            output.append(header)
        counter += 1
    return counter,output


def get_all_params(best_all_params):
    d = {}
    for params in best_all_params.values():
        for param,val in params.items():
            d.setdefault(param,[]).append(val)
    return list(d.values())


def set_directed_G(G):
    G = G.to_directed()
    nx.set_edge_attributes(G,values=1,name='weight')
    G = nx.relabel_nodes(G, lambda x: str(x))
    return G

def get_r_d_k_rho(best_all_params):
    params = list(best_all_params.values())[0]
    return params["r"],params["d"],params["k"],params["rho"]

def plot_rhos(rhos,val_rhos,str_params):
    plt.xlabel("rho")
    plt.xlim(0,1)
    plt.xticks(np.arange(0.0, 1.1, step=0.1))
    plt.ylabel("accuracy")
    plt.title("validation accuracy with different rhos")
    plt.plot(rhos,val_rhos)
    plt.savefig('plots_rho/{}.png'.format(str_params))
    plt.close()


def exe_community(files_name,data_name,CD_model,model_name,has_two_learners,has_communities,is_community_level,seed,
                  only_validate,model_params_path=None,num_to_save=6):
    all_params = convert_file2obj('all_params_community.json')
    threshold = all_params["threshold"]
    to_save = all_params["to_save"]
    to_expand = all_params["to_expand"]
    G = read_gzip_object('{}_{}_{}_{}_{}.gz'.format(data_name,CD_model,int(has_communities),seed,files_name[0]))
    community_dict = read_gzip_object('{}_{}_{}.gz'.format(data_name, CD_model, files_name[1]))
    c_to_partition,num_partitions = None,None
    same_params = all_params["same_params"]
    if only_validate:
        c_to_partition,num_partitions = partition_community_dict(community_dict,G,data_name,CD_model,seed,threshold,to_save)
    else:
        c_to_partition = read_gzip_object('community_to_partition_{}_{}_{}.gz'.format(data_name,CD_model,seed))

    community_dict = {c:[str(u) for u in nodes] for c,nodes in community_dict.items()}
    G = set_directed_G(G)

    #deault p,q,d,r,l,k,workers
    ### d,k and r stay fixed
    rhos = all_params["rhos"]

    # ps, qs, d, r, ls, ks = [0.25, 0.5,2,4], [0.25, 0.5, 2, 4], 64, 16, [30, 50, 70], [5, 7, 12, 16]
    keys = ("p","q","d","r","l","k")
    N = len(keys)

    header,best_params = None,None
    if only_validate:
        groups = [ "Group_{}".format(i) if i >= 0 else "i" for i in range(-1,num_partitions)]
        header = groups + ["rho","Validation accuracy total","params"]
    else:
        best_params = read_csv_params(
            'best_all_params_{}_{}_{}_{}.csv'.format(data_name, CD_model, int(has_communities), seed))
        num_partitions = len(best_params)
        groups = [ "Group_{}".format(i) for i in range(num_partitions)]
        header = groups + ["rho","F1 within","AP within","Accuracy within","F1 between","AP between",
                           "Accuracy between","F1 total","AP total","Accuracy total","params"]

    output = []
    output.append(header)

    print('Node2vec phase (community-wise)')
    counter = 1
    if only_validate:
        dflt_params = [1.0,1.0,128,16,30,12]
        best_params = { i:{keys[n]: dflt_params[n] for n in range(N)}  for i in range(num_partitions)}
        #Note k remains fixed (for all groups) because of context size in Gensim
        # ps, qs, d, r, ls, ks = list(all_params.values())[:6]
        # all_combinations = [(p,q,d,r,l,k) for p in ps for q in qs for l in ls  if l > k]
        all_combinations = [(0.5, 0.25, 128, 16, 5, 3), (0.5, 0.25, 128, 16, 5, 4), (1, 0.25, 128, 16, 5, 3),
                            (1, 0.25, 128, 16, 5, 4), (1, 0.5, 128, 16, 5, 3)
                            ]
        # ,(1, 0.5, 128, 16, 5, 4), (1, 1, 128, 16, 5, 3), (2, 0.25, 128, 16, 5, 3),
        # (2, 0.25, 128, 16, 5, 4), (2, 0.5, 128, 16, 5, 3),
        # (2, 0.5, 128, 16, 5, 4), (2, 1, 128, 16, 5, 3), (2, 1, 128, 16, 5, 4),
        # (4, 0.25, 128, 16, 5, 3), (4, 0.25, 128, 16, 5, 4),
        # (4, 0.5, 128, 16, 5, 3), (4, 0.5, 128, 16, 5, 4), (4, 1, 128, 16, 5, 3),
        # (4, 1, 128, 16, 5, 4), (4, 2, 128, 16, 5, 3),
        # (4, 4, 128, 16, 5, 3)
        ######## NOTE #########
        r,d = 16,128
        iters = num_partitions
        ######## NOTE #########
        if same_params:
            iters = 1
        for i in range(iters):
            print('i = {}'.format(i))
            best_val = 0
            best_rho = 0.0
            best_params_i = best_params[i]
            for combination in all_combinations:
                ######## NOTE #########
                k = combination[-1]
                ######## NOTE #########
                tmp_params_i = {keys[n]: combination[n] for n in range(N)}
                best_params[i] = tmp_params_i
                str_params = str(i)
                if same_params:
                    for j in range(num_partitions):
                        best_params[j] = tmp_params_i
                        str_params = str_params+",".join("{}={}".format(key,val) for key,val in best_params[i].items())
                n2v_params = [i]+[",".join("{}={}".format(key,val) for key,val in params.items()) for params in best_params.values()]
                n2v_params.append(None)
                val_rhos = []
                for rho in rhos:
                    n2v_params[-1] = rho
                    n2v_dict = node2embeddings_community(community_dict,c_to_partition,best_params,G,rho,r,d,k,to_expand)
                    train_dict,test_dict = nodes2edge(n2v_dict,files_name,CD_model,data_name,seed,has_communities,
                                                      is_community_level,to_expand)
                    link_prediction(model_name,has_two_learners,is_community_level,train_dict,test_dict,model_params_path,
                                n2v_params,output,only_validate=True)
                    val_acc = output[-1][-2]
                    val_rhos.append(val_acc)

                    # print('validation accuracy = {}'.format(val_acc))
                    if val_acc > best_val:
                        best_val = val_acc
                        best_params_i = tmp_params_i
                        best_rho = rho
                    if counter % num_to_save == 0:
                        save_res_to_csv(int(counter / num_to_save),data_name,CD_model,has_communities,
                                        is_community_level,has_two_learners,seed,model_name,output)
                        output = []
                        output.append(header)

                    counter += 1
                plot_rhos(rhos,val_rhos,str_params)
            best_params[i] = best_params_i


        lst_best_params = [['p','q','d','r','l','k','rho']]
        if same_params:
            for i in range(num_partitions):
                lst_best_params.append(list(best_params[0].values()))
                lst_best_params[-1].append(best_rho)
        else:
            for i in range(num_partitions):
                lst_best_params.append(list(best_params[i].values()))
                lst_best_params[-1].append(best_rho)

        save_to_csv(lst_best_params, 'best_all_params_{}_{}_{}_{}.csv'.format(data_name,CD_model,int(has_communities),seed))
    else:

        n2v_params =  [",".join("{}={}".format(key,val) for key,val in params.items()) for params in best_params.values()]
        r,d,k,rho = get_r_d_k_rho(best_params)
        n2v_dict = node2embeddings_community(community_dict,c_to_partition,best_params,G,rho,r,d,k)
        train_dict,test_dict = nodes2edge(n2v_dict,files_name,CD_model,data_name,seed,has_communities,
                                          is_community_level,to_expand)
        link_prediction(model_name,has_two_learners,is_community_level,train_dict,test_dict,model_params_path,
                        n2v_params,output,only_validate)

    return counter,output



## l > k , p <= q
#d = 128, r = 10, l = 80, k = 10
# d = 64,128 p,q= 0.25,0.5,1,2  r = 10,12,14  l= 10,30,50,80  k = 10,12,14,16

if __name__ == '__main__':

    input_params = convert_file2obj('input_params.json')

    data_name =input_params["data_name"]
    has_communities = input_params["has_communities"]
    CD_model = input_params["CD_model"]
    if has_communities:
        CD_model = 0
    is_community_level = input_params["is_community_level"]
    has_two_learners = input_params["has_two_learners"]
    model_params_path = input_params["model_params_path"]
    model_name = input_params["model_name"]
    seed = input_params["seed"]
    is_synthetic = input_params["is_synthetic"]
    files_name = ['graph','community_dict','between_dict','train','test','embeddings','log']

    # community_dict = create_dateset(data_name,files_name,CD_model,has_communities=has_communities,is_synthetic=is_synthetic)
    # preprocess_G(data_name,files_name,CD_model,has_communities,community_dict,seed ,frac_within=0.5,frac_between=0.5)

    num_to_save = 3
    counter = 1
    ouptput = None
    if not is_community_level:
        counter,output = exe(files_name,num_to_save,data_name,CD_model,model_name,has_two_learners,has_communities,is_community_level,
                            seed,model_params_path)
    else:
        all_params_community = convert_file2obj('all_params_community.json')
        only_validate = all_params_community["only_validate"]
        counter,output = exe_community(files_name,data_name,CD_model,model_name,has_two_learners, has_communities,
                                       is_community_level, seed,only_validate,model_params_path,num_to_save)

        model_name += '_test'
    print('counter = {}'.format(counter))
    if len(output) > 1:
        if is_community_level:
            data_name = '{}_community'.format(data_name)

        if counter % num_to_save == 0:
            counter = int(counter / num_to_save)
        else:
            counter = int(counter / num_to_save) + 1

        save_res_to_csv(counter,data_name,CD_model,has_communities,is_community_level,
                        has_two_learners,seed,model_name,output)

