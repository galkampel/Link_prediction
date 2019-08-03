
import networkx as nx
from graph import Graph
from preprocess import Preprocess
from node2vec import Node2vec
from model import Model
from edge2vec import *
from community import CommunityNetwork,get_graph_stats
from file2obj import save_to_gzip,read_gzip_object,convert_file2obj,save_to_csv,save_str,save_res_to_csv,read_csv_params
from time import time
import igraph as ig
from itertools import chain,product
from multiprocessing import cpu_count
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



def parallel_node2vec(c,nodes,params,G):#workers = 2,jump = 4)
    G_sub= G.subgraph(nodes)
    G_sub_str = nx.relabel_nodes(G_sub, lambda x: str(x))
    graph = Graph(G=G_sub_str)
    workers = 4#cpu_count()
    # diameter = ig.Graph(edges=list(G_sub.edges()), directed=False).diameter(directed = False)
    # k = diameter + params["k"] * jump
    # print('{},{},{},{},{}'.format(len(G_sub.nodes()),len(G_sub.edges()),
    #     np.mean(list(zip(*G_sub.degree()))[1]),np.median(list(zip(*G_sub.degree()))[1]),diameter))
    node2vec = Node2vec(graph,path_length=params["l"],num_paths=params["r"],
                            p=params["p"],q=params["q"],size=params["d"],window=params["k"],workers=workers)
    vectors = node2vec.get_node2vec()
    return {(c,key):val for key,val in vectors.items()}
    # tmp = {(c,key):val for key,val in vectors.items()}
    # n2v_dict.update(tmp)

def partition_community_dict(data_name,CD_model,has_communities,seed,files_name,thres=10000,to_save=True):
    G = read_gzip_object('{}_{}_{}_{}_{}.gz'.format(data_name,CD_model,int(has_communities),seed,files_name[0]))
    community_dict = read_gzip_object('{}_{}_{}.gz'.format(data_name,CD_model,files_name[1]))
    df = pd.DataFrame.from_dict({key:get_graph_stats(G.subgraph(nodes),count_communities=False) for key,nodes in community_dict.items()},
                                    orient='index',columns = ['nodes','edges','avg degree','med degree','diameter'])

    c_outlies = df[((df['nodes']-df['nodes'].mean()).abs() > 2*df['nodes'].std())].index.tolist()
    rel_df = df[~((df['nodes']-df['nodes'].mean()).abs() > 2*df['nodes'].std())]
    partitions = []
    k_med_dias = tuple(rel_df.groupby(['med degree','diameter']).groups.keys())
    dict_i = {}
    ordered_communities = []
    num_edges = 0
    N = len(k_med_dias)
    for i,k_med_dia in enumerate(k_med_dias):
        k_med,diameter = k_med_dia
        tmp_df = rel_df[(rel_df['med degree'] == k_med) & (rel_df['diameter'] == diameter) ]
        for c in tmp_df.index.tolist():
            dict_i[c] = community_dict[c]
            ordered_communities.append({c:community_dict[c]})
        num_edges += tmp_df['edges'].sum()
        if num_edges > thres:
            num_edges = 0
            partitions.append(dict_i)
            dict_i = {}
        elif i == N-1:
            partitions.append(dict_i)

    for c in c_outlies:
        partitions.append({c: community_dict[c]})
        ordered_communities.append({c:community_dict[c]})

    if to_save:
        df.to_csv("df_{}_{}_{}.csv".format(data_name,CD_model,seed))
        save_to_gzip(ordered_communities,'ordered_communities_{}_{}_{}.gz'.format(data_name,CD_model,seed))

    return partitions


def node2embeddings(params,params_str,data_name,files_name,CD_model,has_communities,seed,is_community_level=False):
    # community_dict = read_gzip_object('{}_{}_{}.gz'.format(data_name,CD_model,files_name[1]))
    print('Node2vec phase (graph-wise)')
    graph = Graph()
    graph.read_edgelist('{}_{}_{}_{}_{}.gz'.format(data_name,CD_model,int(has_communities),seed,files_name[0]))
    start = time()
    node2vec = Node2vec(graph,path_length=params["l"],num_paths=params["r"],
                        p=params["p"],q=params["q"],size=params["d"],window=params["k"],workers=cpu_count())
    print('It took {:.3f} seconds to create nodes representations'.format(time()-start))
    # save_to_gzip(node2vec.get_node2vec(),'{}_{}_{}_{}_{}_{}_{}.gz'.format(data_name,params_str,CD_model,int(has_communities),
    #                                                                    int(is_community_level),seed,files_name[-2]))
    return node2vec.get_node2vec()


def node2embeddings_community(c_partition,params,G):#,to_max = False): #is_community_level=False
    #create a community-level node2vec, and then merge all node2vecs to a single dictionary
    n2v_dict = {}
    # make K classifiers (for each set in the partition (train only on within community links))
    # workers = 4
    # keys = list(params.keys())
    num_partitions = len(c_partition)
    for i in range(num_partitions):
        for c,nodes in c_partition[i].items():
            n2v_dict.update(parallel_node2vec(c,nodes,params[i],G))
    return n2v_dict


def create_train_test(model,data_obj,bin_op,dim,only_validate = False,communities=None):

    if not only_validate:
        data_within_pos = model.edge2vec(list(chain(*data_obj['pos']['within'].values())),bin_op,dim,True)
        data_within_neg = model.edge2vec(list(chain(*data_obj['neg']['within'].values())),bin_op,dim,False)
        data_within = merge_edges(data_within_pos,data_within_neg)
        data_between_pos = model.edge2vec(list(chain(*data_obj['pos']['between'].values())),bin_op,dim,True)
        data_between_neg = model.edge2vec(list(chain(*data_obj['neg']['between'].values())),bin_op,dim,False)
        data_between = merge_edges(data_between_pos,data_between_neg)
        return data_within,data_between
    else:
        # data_within_pos,data_within_neg = None,None
        # if only_validate:
        data_within_pos = model.edge2vec([data_obj['pos']['within'][c] for c in communities.keys()],bin_op,dim,True)
        data_within_neg = model.edge2vec([data_obj['neg']['within'][c] for c in communities.keys()],bin_op,dim,False)
        # else:
        #     data_within_pos = model.edge2vec(list(chain(*data_obj['pos']['within'].values())),bin_op,dim,True)
        #     data_within_neg = model.edge2vec(list(chain(*data_obj['neg']['within'].values())),bin_op,dim,False)
        data_within = merge_edges(data_within_pos,data_within_neg)
        return data_within


def create_train_test_community(model,data_obj,bin_op,dim,only_validate = False,communities=None):

    if not only_validate:
        data_within_pos = model.edge2vec_community(
        list(((c,edge[0]),(c,edge[1])) for c,edges in data_obj['pos']['within'].items() for edge in edges),bin_op,dim,True)
        data_within_neg = model.edge2vec_community(
        list(((c,edge[0]),(c,edge[1])) for c,edges in data_obj['neg']['within'].items() for edge in edges),bin_op,dim,False)
        data_within = merge_edges(data_within_pos,data_within_neg)
        data_between_pos = model.edge2vec_community(
        list(((c1, edge[0]),(c2, edge[1])) for (c1,c2), edges in data_obj['pos']['between'].items() for edge in edges),bin_op,dim,True)
        data_between_neg = model.edge2vec_community(
        list(((c1, edge[0]), (c2, edge[1])) for (c1, c2), edges in data_obj['neg']['between'].items() for edge in edges) ,bin_op,dim,False)
        data_between = merge_edges(data_between_pos,data_between_neg)
        return data_within,data_between
    else:
        # data_within_pos,data_within_neg = None,None
        # if only_validate:
        data_within_pos = model.edge2vec_community(
        list(((c, edge[0]), (c, edge[1])) for c in communities.keys() for edge in data_obj['pos']['within'][c] ), bin_op, dim, True)
        data_within_neg = model.edge2vec_community(
        list(((c, edge[0]), (c, edge[1])) for c in communities.keys() for edge in data_obj['neg']['within'][c] ),bin_op,dim,False)
        # else:
        #     data_within_pos = model.edge2vec(list(chain(*data_obj['pos']['within'].values())),bin_op,dim,True)
        #     data_within_neg = model.edge2vec(list(chain(*data_obj['neg']['within'].values())),bin_op,dim,False)
        data_within = merge_edges(data_within_pos,data_within_neg)
        return data_within



#create 2 datasets: one for within edges, and the other for between edges
#using Hadamard because it was the best binary operator according to the article



def nodes2edge(vectors,files_name,CD_model,data_name,seed,has_communities): #params_str

    train_obj = read_gzip_object('{}_{}_{}_{}_train.gz'.format(data_name,CD_model,int(has_communities),seed,files_name[3]))
    test_obj = read_gzip_object('{}_{}_{}_{}_{}.gz'.format(data_name,CD_model,int(has_communities),seed,files_name[4]))
    # vectors = read_gzip_object('{}_{}_{}_{}_{}_{}_{}.gz'.format(data_name,params_str,CD_model,int(has_communities),
    #                                                                        int(is_community_level),seed,files_name[-2]))
    print("start converting nodes to edges")
    d = list(vectors.values())[0].shape[0]
    model = Edge2vec(vectors)
    start = time()
    # if not only_validate:
        ####### ADDITION ##########
        # community_dict = read_gzip_object('{}_{}_{}.gz'.format(data_name, CD_model, files_name[1]))
        # train_within, train_between = create_train_test(model, train_obj, Hadamard, d, communities=community_dict)
        # test_within, test_between = create_train_test(model, test_obj, Hadamard, d, communities=community_dict)
        ####### ADDITION ##########
    train_within,train_between = create_train_test(model,train_obj,Hadamard,d)
    test_within,test_between = create_train_test(model,test_obj,Hadamard,d)

    print('It takes {:.3f} seconds to create edge embeddings'.format(time() - start))

        # save 4 datasets (within/between for train/test)
    train = {'within': train_within,'between':train_between}
    test = {'within': test_within,'between':test_between}
    return train,test
    # else:
    #     train_within = create_train_test(model,train_obj,Hadamard,d,only_validate,communities)
    #     test_within = create_train_test(model,test_obj,Hadamard,d,only_validate,communities)
    #
    #     print('It takes {:.3f} seconds to create edge embeddings'.format(time() - start))
    #     # save 2 datasets (within for train/test)
    #     train = {'within': train_within}
    #     test = {'within': test_within}
    #     return train,test


    # save_to_gzip(train,files_name[3]+'{}_{}_{}.gz'.format(seed,type,id))
    # save_to_gzip(test,files_name[4]+'{}_{}_{}.gz'.format(seed,type,id))

def nodes2edge_community(vectors,files_name,CD_model,data_name,seed,has_communities,only_validate=False,communities=None):
    train_obj = read_gzip_object('{}_{}_{}_{}_train.gz'.format(data_name,CD_model,int(has_communities),seed,files_name[3]))
    test_obj = read_gzip_object('{}_{}_{}_{}_{}.gz'.format(data_name,CD_model,int(has_communities),seed,files_name[4]))
    # vectors = read_gzip_object('{}_{}_{}_{}_{}_{}_{}.gz'.format(data_name,params_str,CD_model,int(has_communities),
    #                                                                        int(is_community_level),seed,files_name[-2]))
    print("start converting nodes to edges")
    d = list(vectors.values())[0].shape[0]

    model = Edge2vec(vectors)
    start = time()
    if not only_validate:
        train_within,train_between = create_train_test_community(model,train_obj,Hadamard,d) #communities=communities
        test_within,test_between = create_train_test_community(model,test_obj,Hadamard,d) #communities=communities

        print('It takes {:.3f} seconds to create edge embeddings'.format(time() - start))
        # save 4 datasets (within/between for train/test)
        train = {'within': train_within,'between':train_between}
        test = {'within': test_within,'between':test_between}
        return train,test
    else:
        train_within = create_train_test_community(model,train_obj,Hadamard,d,only_validate,communities)
        test_within = create_train_test_community(model,test_obj,Hadamard,d,only_validate,communities)

        print('It takes {:.3f} seconds to create edge embeddings'.format(time() - start))
        # save 2 datasets (within for train/test)
        train = {'within': train_within}
        test = {'within': test_within}
        return train,test



#save csv file where each row is comprised from:
    # node2vec parameters: p,q,d,r,l,k
    # model name (logistic regression, MLP, xgboost or Random Forest)
    # evaluation measures: acc?, F1, AP X3 (between,within,total)

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
    all_params["workers"] = cpu_count()
    output = []
    header = ["p","q","d","r","l","k","F1 within","AP within","Accuracy within",
              "F1 between","AP between","Accuracy between","F1 total","AP total","Accuracy total","params"]
    output.append(header)
    counter = 1
    np.random.seed(1)
    # all_combinations = list(product(all_params["p"],all_params["q"],all_params["d"],all_params["r"],all_params["l"],all_params["k"]))
    all_combinations = [(0.5, 0.25, 128, 16, 5, 3),(0.5,0.25,128,16,5,4),(1,0.25,128,16,5,3),(1,0.25,128,16,5,4),(1,0.5,128,16,5,3),
                        (1, 0.5, 128, 16, 5, 4),(1,1,128,16,5,3),(2,0.25,128,16,5,3),(2,0.25,128,16,5,4),(2,0.5,128,16,5,3),
                        (2, 0.5, 128, 16, 5, 4),(2,1,128,16,5,3),(2,1,128,16,5,4),(4,0.25,128,16,5,3),(4,0.25,128,16,5,4),
                        (4, 0.5, 128, 16, 5, 3),(4,0.5,128,16,5,4),(4,1,128,16,5,3),(4,1,128,16,5,4),(4,2,128,16,5,3),
                        (4, 4, 128, 16, 5, 3)]
    # p_qs = [(all_params["p"][i], all_params["q"][j]) for i in range(len(all_params["p"])) for j in range(i, len(all_params["q"]))]
    # all_combinations = list(map(lambda el: el[0] + el[1], list(product(p_qs, list(product(all_params["d"], all_params["r"], all_params["l"],all_params["k"]))))))
    # idxes = np.random.choice(range(len(all_combinations)),num_combinations,replace = False)
    # cand_params = [all_combinations[idx] for idx in idxes]
    workers = all_params["workers"]
    # for cand_param in cand_params:
    for cand_param in all_combinations:
        p,q,d,r,l,k = cand_param
        params = {"p": p,"q": q, #"p": all_params["p"][arg_p],"q": all_params["q"][arg_q]
          "d":d,"r":r,"l":l,"k":k,"workers":workers}
        n2v_params = list(params.values())[:-1]
        print("Node2vec parameters: {}".format(params))
        params_str = "_".join("{}={}".format(key,val) for key,val in params.items() if key !='workers')
        n2v_dict = node2embeddings(params,params_str,data_name,files_name,CD_model,has_communities,seed,is_community_level)
        print('finished creating node2vec')
        train_dict,test_dict = nodes2edge(n2v_dict,files_name,CD_model,data_name,seed,has_communities)
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


def get_G(data_name,CD_model,has_communities,seed,files_name):
    G = read_gzip_object('{}_{}_{}_{}_{}.gz'.format(data_name,CD_model,int(has_communities),seed,files_name[0]))
    G = G.to_directed()
    nx.set_edge_attributes(G,values=1,name='weight')
    return G

def exe_community(files_name,data_name,CD_model,model_name,has_two_learners,has_communities,is_community_level,seed,
                  model_params_path=None,num_to_save=120):
    all_params = convert_file2obj('all_params_community.json')
    threshold = all_params["threshold"]
    to_save = all_params["to_save"]
    c_partition = partition_community_dict(data_name,CD_model,has_communities,seed,files_name,threshold,to_save)
    #deault p,q,d,r,l,k,workers
    num_partition = len(c_partition)
    # ps,qs,d,r,ls,ks = [0.25,0.5,1,2,4], [0.25,0.5,1,2,4],64,16,[30,50,70],[5,7,9,12]
    ps = all_params["p"]
    qs = all_params["q"]
    d = all_params["d"]
    r = all_params["r"]
    ls = all_params["l"]
    ks = all_params["k"]
    # ps, qs, d, r, ls, ks = [0.25, 0.5,2,4], [0.25, 0.5, 2, 4], 64, 16, [30, 50, 70], [5, 7, 12, 16]
    all_combinations = [(p,q,d,r,l,k) for p in ps for q in qs for l in ls for k in ks if l > k]
    keys = ("p","q","d","r","l","k")
    N = len(keys)
    header_val = ["Group","p","q","d","r","l","k","Accuracy within","params"]
    output_val = []
    output_val.append(header_val)
    counter = 1
    best_params = []
    G = get_G(data_name,CD_model,int(has_communities),seed,files_name)
    print('Node2vec phase (community-wise)')
    for i in range(num_partition):
        print('i = {}'.format(i))
        best_val = 0
        best_params_i = {}
        for combination in all_combinations:
            params_partition = {keys[n]: combination[n] for n in range(N)}
            n2v_params =  [i]+list(params_partition.values())
            n2v_dict = node2embeddings_community([{c:nodes for c,nodes in c_partition[i].items()}],
                                                 [params_partition],G)
            train_dict,test_dict = nodes2edge_community(n2v_dict,files_name,CD_model,data_name,seed,has_communities,
                                                        True,c_partition[i])
            tmp_has_two_learners = False
            link_prediction(model_name,tmp_has_two_learners,is_community_level,train_dict,test_dict,model_params_path,
                            n2v_params,output_val,only_validate=True)
            val_acc = output_val[-1][-2]
            if val_acc > best_val:
                best_val = val_acc
                best_params_i = params_partition
            if counter % num_to_save == 0:
                save_res_to_csv(int(counter / num_to_save),data_name,CD_model,has_communities,
                                is_community_level,has_two_learners,seed,model_name,output_val)
                output_val = []
                output_val.append(header_val)
            counter += 1


        best_params.extend([best_params_i]*len(c_partition[i]))

    lst_best_params = [['p','q','d','r','l','k']]
    for params in best_params:
        lst_best_params.append(list(params.values()))

    save_to_csv(lst_best_params, 'best_all_params_{}_{}_{}_{}.csv'.format(data_name,CD_model,int(has_communities),seed))
            #train for each combination. if improvement in validation -> change best_paramas_i
    return counter,output_val



def exe_community_test(files_name,data_name,CD_model,model_name,has_communities,is_community_level,seed,
                  model_params_path=None):

    ordered_communities = read_gzip_object('ordered_communities_{}_{}_{}.gz'.format(data_name,CD_model,seed))
    best_all_params = read_csv_params('best_all_params_{}_{}_{}_{}.csv'.format(data_name,CD_model,int(has_communities),seed))

    output = []
    header = ["F1 within","AP within","Accuracy within",
              "F1 between","AP between","Accuracy between","F1 total","AP total","Accuracy total","params"]
    output.append(header)
    counter = 1
    G = get_G(data_name,CD_model,int(has_communities),seed,files_name)
    print('Node2vec test phase (community-wise) ')
    n2v_dict = node2embeddings_community(ordered_communities,best_all_params,G)
    train_dict,test_dict = nodes2edge_community(n2v_dict,files_name,CD_model,data_name,seed,has_communities,
                                                False,ordered_communities)
    link_prediction(model_name,has_two_learners,is_community_level,train_dict,test_dict,model_params_path,n2v_params= [],output=output)
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

    num_to_save = 2
    counter = 0
    ouptput = None
    if not is_community_level:
        counter,output = exe(files_name,num_to_save,data_name,CD_model,model_name,has_two_learners,has_communities,is_community_level,
                            seed,model_params_path)
    else:
        all_params_community = convert_file2obj('all_params_community.json')
        only_validate = all_params_community["only_validate"]
        if only_validate:
            num_to_save = 360
            counter,output = exe_community(files_name,data_name,CD_model,model_name,has_two_learners,
                                           has_communities,is_community_level, seed,model_params_path,num_to_save)
        else:
            counter,output = exe_community_test(files_name,data_name,CD_model,model_name,has_communities,
                                                is_community_level, seed,model_params_path)
            model_name += '_test'
    print('counter = {}'.format(counter))
    if len(output) > 1:
        save_res_to_csv(int(counter / num_to_save) + 1,data_name,CD_model,has_communities,
                                                is_community_level,has_two_learners,seed,model_name,output)

