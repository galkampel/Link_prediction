
# import networkx as nx
from graph import Graph
from preprocess import Preprocess
# from node2vec import Node2vec
from walker import DeepWalker #,expand_node
from nodeEmbeddings import *
from model import Model
from edge2vec import *
from splitter import Splitter
from community import CommunityNetwork #,get_graph_stats
from file2obj import save_to_gzip,read_gzip_object,convert_file2obj,save_str,save_res_to_csv #,save_vec_to_HDF5,read_vec_from_HDF5,save_to_csv,read_csv_params
from time import time
from itertools import chain#,product
from multiprocessing import cpu_count
from gensim.models import Word2Vec
# import matplotlib.pyplot as plt
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

        return


#create 'a given' graph to predict from
def preprocess_G(data_name,files_name,CD_model,has_communities,community_dict,multiplier,seed = 1,frac_within=0.5,frac_between=0.5):
    G = read_gzip_object('{}_{}_{}.gz'.format(data_name,CD_model,files_name[0]))
    # community_dict = read_gzip_object('{}_{}_{}.gz'.format(data_name,CD_model,files_name[1]))
    between_edges_dict = read_gzip_object('{}_{}_{}.gz'.format(data_name,CD_model,files_name[2]))
    print('Start preprocessing...')
    #default seed = 1
    preprocess = Preprocess(G,community_dict,between_edges_dict,multiplier,frac_within,frac_between,seed = seed)
    start = time()
    preprocess.set_new_G('{}_{}_{}_{}_{}'.format(data_name,CD_model,int(has_communities),seed,files_name[4]))
    print('It took {:.3f} seconds to create new G and test set'.format(time()-start))
    start = time()
    preprocess.save_train_obj('{}_{}_{}_{}_{}'.format(data_name,CD_model,int(has_communities),seed,files_name[3]))
    print('It took {:.3f} seconds to create new training set'.format(time()-start))
    save_to_gzip(preprocess.get_G(),'{}_{}_{}_{}_{}.gz'.format(data_name,CD_model,int(has_communities),seed,files_name[0]))


# def partition_community_dict(community_dict,G,data_name,CD_model,seed,thres=10000,to_save=True):
#
#     df = pd.DataFrame.from_dict({key:get_graph_stats(G.subgraph(nodes),count_communities=False) for key,nodes in community_dict.items()},
#                                     orient='index',columns = ['nodes','edges','avg degree','r_avg degree','med degree'])
#     is_outlier = (((df['r_avg degree'] - df['r_avg degree'].mean()).abs() > 2 * df['r_avg degree'].std())
#                     & ((df['r_avg degree'] - df['r_avg degree'].mean()).abs() > 2 * df['r_avg degree'].std()))
#     c_outlies = df[is_outlier].index.tolist()
#     rel_df = df[~is_outlier]
#     k_rounded_meds = tuple(rel_df.groupby(['r_avg degree', 'med degree']).groups.keys())
#     c_to_parj = {}
#     num_edges = 0
#     N = len(k_rounded_meds)
#     j = 0
#     for i,k_rounded_med in enumerate(k_rounded_meds):
#         k_rounded,k_med = k_rounded_med
#         tmp_df = df[(df['r_avg degree'] == k_rounded) & (df['med degree'] == k_med) ]
#         for c in tmp_df.index.tolist():
#             c_to_parj[c] = j
#
#         num_edges += tmp_df['edges'].sum()
#         if num_edges > thres or i == N-1:
#             j += 1
#             num_edges = 0
#
#     for c in c_outlies:
#         c_to_parj[c] = j
#         j += 1
#
#     if to_save:
#         df.to_csv("df_{}_{}_{}.csv".format(data_name,CD_model,seed))
#         save_to_gzip(c_to_parj,'community_to_partition_{}_{}_{}.gz'.format(data_name,CD_model,seed))
#
#     return c_to_parj,j+1


# def node2embeddings(params,data_name,files_name,CD_model,has_communities,community_dict,multiplier,seed):
#     # community_dict = read_gzip_object('{}_{}_{}.gz'.format(data_name,CD_model,files_name[1]))
#     print('Node2vec phase (graph-wise)')
#     graph = Graph()
#     graph.read_edgelist('{}_{}_{}_{}_{}.gz'.format(data_name,CD_model,int(has_communities),seed,files_name[0]))
#     start = time()
#     node2vec = Node2vec(graph.G,params,community_dict,multiplier,size=params["d"],window=params["k"],
#                         workers=cpu_count())
#     print('It took {:.3f} seconds to create nodes representations'.format(time()-start))
#     # save_to_gzip(node2vec.get_node2vec(),'{}_{}_{}_{}_{}_{}_{}.gz'.format(data_name,params_str,CD_model,int(has_communities),
#     return node2vec.get_node2vec()

# def node2embeddings_community(community_dict,G,params,multiplier,to_expand):#,to_max = False): #is_community_level=False
#     #create a community-level node2vec, and then merge all node2vecs to a single dictionary
#     # graph = Graph()
#     # graph.read_edgelist('{}_{}_{}_{}_{}.gz'.format(data_name,CD_model,int(has_communities),seed,files_name[0]))
#     start = time()
#     d,k = params["d"],params["k"]
#     node2vec = Node2vec(G,params,community_dict,multiplier,is_community_level = True,
#                         to_expand=to_expand, size=d,window=k,workers=cpu_count())
#     print('It took {:.3f} seconds to create nodes representations'.format(time()-start))
#     return node2vec.get_node2vec()


# def edge2community_edge(data,is_within):
#     if is_within:
#         within_pos_edges = list((expand_node(link[0],c),expand_node(link[1],c) )
#                                 for c,links in data['pos']['within'].items() for link in links)
#         within_neg_edges = list((expand_node(link[0],c),expand_node(link[1],c) )
#                                 for c,links in data['neg']['within'].items() for link in links)
#         return within_pos_edges,within_neg_edges
#     else:
#         between_pos_edges = list((expand_node(link[0],c1),expand_node(link[1],c2) )
#                                 for (c1,c2),links in data['pos']['between'].items() for link in links)
#         between_neg_edges = list((expand_node(link[0],c1),expand_node(link[1],c2) )
#                                 for (c1,c2),links in data['neg']['between'].items() for link in links)
#         return between_pos_edges,between_neg_edges


def create_train_test(model,data_obj,bin_op,dim): #,is_community_level = False,to_expand = False

    # if is_community_level and to_expand:
    #     within_pos_edges,within_neg_edges = edge2community_edge(data_obj,is_within=True)
    #     between_pos_edges,between_neg_edges = edge2community_edge(data_obj,is_within=False)
    #     data_within_pos = model.edge2vec(within_pos_edges,bin_op,dim,True)
    #     data_within_neg = model.edge2vec(within_neg_edges,bin_op,dim,False)
    #     data_within = merge_edges(data_within_pos,data_within_neg)
    #     data_between_pos = model.edge2vec(between_pos_edges,bin_op,dim,True)
    #     data_between_neg = model.edge2vec(between_neg_edges,bin_op,dim,False)
    #     data_between = merge_edges(data_between_pos,data_between_neg)
    #     return data_within,data_between
    # else:
    data_within_pos = model.edge2vec(list(chain(*data_obj['pos']['within'].values())),bin_op,dim,True)
    data_within_neg = model.edge2vec(list(chain(*data_obj['neg']['within'].values())),bin_op,dim,False)
    data_within = {'pos':data_within_pos,'neg': data_within_neg}
    data_between_pos = model.edge2vec(list(chain(*data_obj['pos']['between'].values())),bin_op,dim,True)
    data_between_neg = model.edge2vec(list(chain(*data_obj['neg']['between'].values())),bin_op,dim,False)
    data_between = {'pos':data_between_pos,'neg': data_between_neg}
    return data_within,data_between


#create 2 datasets: one for within edges, and the other for between edges
#using Hadamard because it was the best binary operator according to the article

def nodes2edge(vectors,files_name,CD_model,data_name,seed,method,
               has_communities,to_diff_links,save_edge_embeddings): #params_str, to_expand=False

    train_obj = read_gzip_object('{}_{}_{}_{}_{}.gz'.format(data_name,CD_model,int(has_communities),seed,files_name[3]))
    test_obj = read_gzip_object('{}_{}_{}_{}_{}.gz'.format(data_name,CD_model,int(has_communities),seed,files_name[4]))
    # vectors = read_gzip_object('{}_{}_{}_{}_{}_{}_{}.gz'.format(data_name,params_str,CD_model,int(has_communities),
    #                                                                        int(is_community_level),seed,files_name[-2]))
    print("start converting nodes to edges")
    d = list(vectors.values())[0].shape[0]
    model = Edge2vec(vectors)
    start = time()
    train_within,train_between = create_train_test(model,train_obj,Hadamard,d)
    test_within,test_between = create_train_test(model,test_obj,Hadamard,d)

    print('It takes {:.3f} seconds to create edge embeddings'.format(time() - start))
    train = {'within': train_within, 'between': train_between}
    test = {'within': test_within, 'between': test_between}
        # save 4 datasets (within/between for train/test)
    if to_diff_links and save_edge_embeddings:
        #create a link embeddings using a classifier and saving sate dictionary of the embeddings network
        splitter = Splitter(method)
        splitter.set_within_between_sep(train_within['pos'],train_between['pos'])
    return train,test


def link_prediction(model_name,to_diff_links,is_multiclass,method,seed,
                    train_dict,test_dict,model_params_path ,params_str,output):
    model_params = convert_file2obj(model_params_path)
    model = Model(model_params,to_diff_links,is_multiclass,method,seed)
    if model_name == 'Logistic Regression' or model_name == 'xgboost':
        model.get_measures(train_dict,test_dict,output,model_name,params_str)
    else:
        model.get_MLP_measures(train_dict,test_dict,output,params_str)

#has_communities:
# if 0 then need to use community detection algorithm to create communities
#CD_model:
# 0 if no need of CD algorithm, 1 for multilevel, 2 for infomap and 3 for label propagation


#is_community_level:
# 1 if community-level based node2vec should be created else (0) graph-level

# two types of prediction:
# type '1' - ignore community knowledge
# type '2' - differentiate between within and between community links
# def exe_n2v(files_name,num_to_save,data_name,CD_model,model_name,to_diff_links,is_multiclass,
#         has_communities, multiplier,seed,seed_trainings,all_params,model_params_path=None):
#
#     # all_params = convert_file2obj('all_params_tmp.json')
#     all_params["workers"] = cpu_count()
#     output = []
#     header = ["p","q","d","r","l","k","F1 within","AP within","Accuracy within","AUC within","F1 between","AP between",
#               "Accuracy between","AUC between","F1 total","AP total","Accuracy total","AUC total","params"]
#     output.append(header)
#     counter = 1
#     pqs, d, r, ls, ks = list(all_params.values())[:5]
#     to_read_n2v_dict = all_params["to_read_n2v_dict"]
#     to_save_n2v_dict = all_params["to_save_n2v_dict"]
#     all_combinations = [(pq[0],pq[1],d,r,l,k) for pq in pqs for l in ls for k in ks  if l > k]
#     community_dict = read_gzip_object('{}_{}_{}.gz'.format(data_name, CD_model, files_name[1]))
#     community_dict = {c:[str(u) for u in nodes] for c,nodes in community_dict.items()}
#     for seed_training in seed_trainings:
#         for cand_param in all_combinations:
#             p,q,d,r,l,k = cand_param
#             params = {"p": p,"q": q, "d":d,"r":r,"l":l,"k":k,"workers":all_params["workers"]}
#             n2v_params = list(params.values())[:-1]
#             print("Node2vec parameters: {}".format(params))
#             params_str = "_".join("{}={}".format(key,val) for key,val in params.items() if key !='workers')
#             n2v_dict = None
#             if to_read_n2v_dict:
#                 n2v_dict = read_gzip_object('n2v_dict_{}_{}_{}_{}.gz'.format(data_name,params_str,int(has_communities),seed))
#             else:
#                 n2v_dict = node2embeddings(params,data_name,files_name,CD_model,has_communities,community_dict,multiplier,seed)
#                 if to_save_n2v_dict:
#                     save_to_gzip(n2v_dict,'n2v_dict_{}_{}_{}_{}.gz'.format(data_name,params_str,int(has_communities),seed))
#             print('finished creating node2vec')
#
#
#             train_dict,test_dict = nodes2edge(n2v_dict,files_name,CD_model,data_name,seed,has_communities)
#             link_prediction(model_name,to_diff_links,is_multiclass,seed_training,
#                             train_dict,test_dict,model_params_path,n2v_params,output)
#             if counter % num_to_save == 0:
#                 save_res_to_csv(int(counter / num_to_save),data_name,CD_model,has_communities,
#                                 to_diff_links,is_multiclass,seed,seed_training,model_name,output)
#                 output = []
#                 output.append(header)
#             counter += 1
#         if len(output) > 1:
#             if counter % num_to_save == 0:
#                 counter = int(counter / num_to_save)
#             else:
#                 counter = int(counter / num_to_save) + 1
#             save_res_to_csv(counter,data_name,CD_model,has_communities,to_diff_links,
#                             is_multiclass,seed,seed_training,model_name,output)
#         counter = 1
#         output = []
#         output.append(header)
#     return

def config_to_str(config):
    str_config = "("
    N = len(config)
    for i in range(N):
        if i < N-1:
            str_config +="{},".format(config[i])
        else:
            str_config +="{}".format(config[i])
    str_config += ")"
    return str_config

def print_num_good_links(G,links,type):
    N = len(links)
    num_good_links = 0
    for link in links:
        if len(set(G.nodes[link[0]]['label']).intersection(set(G.nodes[link[1]]['label']))) == 0:
            num_good_links += 1
    print('{:.3f} of the {} between edges are good'.format((num_good_links/N),type))

def check_between_edges(data_name,CD_model,has_communities,seed,files_name):
    between_edges_dict = read_gzip_object('{}_{}_{}.gz'.format(data_name,CD_model,files_name[2]))
    G = read_gzip_object('{}_{}_{}_{}_{}.gz'.format(data_name, CD_model, int(has_communities), seed, files_name[0]))
    between_train_edges = list(chain(*between_edges_dict.values()))
    print_num_good_links(G,between_train_edges,'train')
    test_obj = read_gzip_object('{}_{}_{}_{}_{}.gz'.format(data_name,CD_model,int(has_communities),seed,files_name[4]))
    between_test_edges = list(chain(*test_obj['pos']['between'].values()))
    print_num_good_links(G,between_test_edges,'test')


def exe(files_name,method,num_to_save,data_name,CD_model,model_name,to_diff_links,save_edge_embeddings,is_multiclass,
        has_communities, multiplier,seed,seed_trainings,n2embedd_params,model_params_path):

    output = []
    header = ["Method","Method_params","F1 within", "AP within", "Accuracy within",
              "AUC within", "F1 between","AP between","Accuracy between", "AUC between",
              "F1 total", "AP total","Accuracy total", "AUC total", "params"]
    # to_read_embed_dict = all_params["to_read_embed_dict"]
    has_embedded_dict = n2embedd_params["has_embedded_dict"]
    output.append(header)
    community_dict = read_gzip_object('{}_{}_{}.gz'.format(data_name, CD_model, files_name[1]))
    community_dict = {c: [str(u) for u in nodes] for c, nodes in community_dict.items()}
    graph = Graph()
    graph.read_edgelist('{}_{}_{}_{}_{}.gz'.format(data_name, CD_model, int(has_communities), seed, files_name[0]))

    #which node2embedding to execute
    node_embeddings = None
    counter = 1
    config_name,configs = None,None
    # check_between_edges(data_name,CD_model,has_communities,seed,files_name)
    params_str,tmp_params_str = "",""
    if method == "ELMo":
        node_embeddings = get_ElmoEmbeddings
        configs = n2embedd_params["scalar_mix_parameters"]
        config_name = "scalar_mix_parameters"
        tmp_params_str = "_".join( "{}={}".format(key, val) for key, val in n2embedd_params.items()
                    if key != config_name and key != "sentences" and key != "nodes2idx" and key != "has_embedded_dict")
        if not has_embedded_dict:
            set_sentences_node2idx(graph.G,n2embedd_params,seed)
    elif method == "DeepWalk":
        node_embeddings = get_DeepWalk
        configs = n2embedd_params["alphas"]
        config_name = "alpha"
        tmp_params_str = "_".join( "{}={}".format(key, val) for key, val in n2embedd_params.items()
                    if key != "{}s".format(config_name) and key != "has_embedded_dict")

    elif method == "ComE":
        2
        #ComE
    else:
        #M-NMF
        23
    for seed_training in seed_trainings:
        for config in configs:

            if method == "ELMo":
                params_str = "{}_{}={}".format(tmp_params_str,config_name,config_to_str(config))
            elif method == "DeepWalk":
                params_str = "{}_{}={}".format(tmp_params_str,config_name,str(config))
            pre_measures = [method,params_str]
            vectors = None
            if not has_embedded_dict:
                start = time()
                vectors = node_embeddings(graph.G,n2embedd_params,config)
                print('it took {:.3f} seconds to create node embeddings dict'.format(time()-start))
                save_to_gzip(vectors,'vectors_{}_{}.gz'.format(method,params_str))
                print('save nodes representation dict\nContiue..')
                continue
            else: #has_embedded_dict
                vectors = read_gzip_object('vectors_{}_{}.gz'.format(method,params_str))
            train_dict, test_dict = nodes2edge(vectors, files_name, CD_model, data_name, seed, method,
                                               has_communities, to_diff_links,save_edge_embeddings)

            link_prediction(model_name, to_diff_links, is_multiclass,method, seed_training,
                            train_dict, test_dict, model_params_path, pre_measures, output)
            if counter % num_to_save == 0:
                save_res_to_csv(int(counter / num_to_save), data_name, CD_model, has_communities,
                                to_diff_links, is_multiclass, seed, seed_training, model_name, output)
                output = []
                output.append(header)
            counter += 1
        if has_embedded_dict:
            print('finished looping over configs.\nStop execution')
            exit()
        if len(output) > 1:
            if counter % num_to_save == 0:
                counter = int(counter / num_to_save)
            else:
                counter = int(counter / num_to_save) + 1
            save_res_to_csv(counter, data_name, CD_model, has_communities, to_diff_links,
                            is_multiclass, seed, seed_training, model_name, output)
    counter = 1
    output = []
    output.append(header)

def set_sentences_node2idx(G,params,seed):
    walker = DeepWalker(G)
    start = time()
    walker.preprocess_transition_probs()
    print('it took {:.3f} seconds to create transition probs'.format(time()-start))
    sentences, nodes2idx = walker.simulate_walks(params["num_walks"],params["walk_length"])
    save_to_gzip(sentences,'sentences_l={}_r={}.gz'.format(params["walk_length"],params["num_walks"]))
    save_to_gzip(nodes2idx,'nodes2idx_l={}_r={}.gz'.format(params["walk_length"],params["num_walks"]))
    exit()


def get_ElmoEmbeddings(G,params,scalar_mix_parameters):

    elmo = ElmoEmbeddings(scalar_mix_parameters,params["num_output_representations"],params["dropout"],
                          params["requires_grad"])
    elmo.set_ELMO_embeddings(params["sentences"])
    print('finished creating embeddings')
    return elmo.get_vectors(params["nodes2idx"])

def DW_representation(sentences,params,config):
    kwargs = {}
    kwargs["workers"] = cpu_count()
    kwargs["size"] = params["d"]
    kwargs["window"] = params["num_walks"]
    kwargs["sentences"] = sentences
    kwargs["min_count"] = 0
    kwargs["sg"] = 1
    kwargs["alpha"] = config #best 0.15
    # kwargs["compute_loss"] = True
    # kwargs["iter"] = 5 #100
    return Word2Vec(**kwargs)



def get_DeepWalk(G,params,config): #config = alpha
    dw = DeepWalker(G)
    dw.preprocess_transition_probs()
    sentences = dw.simulate_walks(params["num_walks"],params["walk_length"],calc_nodes2idx=False)
    word2vec = DW_representation(sentences,params,config)
    vectors = {}
    for word in G.nodes():
        vectors[word] = word2vec[word]
    return vectors



# def get_all_params(best_all_params):
#     d = {}
#     for params in best_all_params.values():
#         for param,val in params.items():
#             d.setdefault(param,[]).append(val)
#     return list(d.values())


# def set_directed_G(G):
#     G = G.to_directed()
#     nx.set_edge_attributes(G,values=1,name='weight')
#     G = nx.relabel_nodes(G, lambda x: str(x))
#     return G
#
# def get_r_d_k_rho(best_all_params):
#     params = list(best_all_params.values())[0]
#     return params["r"],params["d"],params["k"],params["rho"]
#
# def plot_rhos(rhos,val_rhos,params_str):
#     plt.xlabel("rho")
#     plt.xlim(0,1)
#     plt.xticks(np.arange(0.0, 1.1, step=0.1))
#     plt.ylabel("accuracy")
#     plt.title("validation accuracy with different rhos")
#     plt.plot(rhos,val_rhos)
#     plt.savefig('plots_rho/{}.png'.format(params_str))
#     plt.close()


## l > k , p <= q
#d = 128, r = 10, l = 80, k = 10
# d = 64,128 p,q= 0.25,0.5,1,2  r = 10,12,14  l= 10,30,50,80  k = 10,12,14,16

if __name__ == '__main__':


    input_params = convert_file2obj('input_params.json')
    data_name =input_params["data_name"]
    has_communities = input_params["has_communities"]
    CD_model = input_params["CD_model"]
    multiplier = 1
    if data_name == "Flickr":
        multiplier = 5
    if has_communities:
        CD_model = 0
    all_params = None
    # to_expand = None
    # to_intertwine = None


    to_diff_links = input_params["to_diff_links"]
    save_edge_embeddings = input_params["save_edge_embeddings"]
    is_multiclass = input_params["is_multiclass"]
    model_params_path = input_params["model_params_path"]
    method = input_params["method"]
    all_params = convert_file2obj('{}_params.json'.format(method))
    model_name = input_params["model_name"]
    seed = input_params["seed"]
    seed_trainings = input_params["seed_trainings"]
    is_synthetic = input_params["is_synthetic"]
    has_community_dict = input_params["has_community_dict"]
    files_name = ['graph','community_dict','between_dict','train','test','embeddings','log']

    # if not has_community_dict:
    #     create_dateset(data_name,files_name,CD_model,has_communities=has_communities,is_synthetic=is_synthetic)
    #     exit()
    # else:
    #     community_dict = read_gzip_object('{}_{}_{}.gz'.format(data_name, CD_model, files_name[1]))
    #     preprocess_G(data_name,files_name,CD_model,has_communities,community_dict,multiplier,seed,frac_within=0.5,frac_between=0.5)
    #     print('Finish preprocessing.\nQuit execution!!')
    #     exit()

    num_to_save = 1
    counter = 1
    # ouptput = None
    exe(files_name,method,num_to_save,data_name,CD_model,model_name,to_diff_links,save_edge_embeddings,is_multiclass,
        has_communities,multiplier,seed,seed_trainings,all_params,model_params_path)

