
from preprocess import Preprocess
from feature import Features
from model import Model
from sklearn.preprocessing import MinMaxScaler
from edge2vec import *
from community import CommunityNetwork
from file2obj import save_to_gzip,read_gzip_object,convert_file2obj,save_str,save_res_to_csv,save_to_csv
from time import time
from itertools import chain
import gc
gc.collect()



# create/import dataset.
#If has no communities, communities are needed to be completed


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



def create_feature_train_test(data_obj,model,type):

    data_within_pos = model.create_dataset(list(chain(*data_obj['pos']['within'].values())),1)
    data_within_neg = model.create_dataset(list(chain(*data_obj['neg']['within'].values())),0)
    data_within = merge_edges(data_within_pos,data_within_neg)
    data_between_pos = model.create_dataset(list(chain(*data_obj['pos']['between'].values())),1)
    data_between_neg = model.create_dataset(list(chain(*data_obj['neg']['between'].values())),0)

    # print('within pos = {}   within neg = {}\nbetween pos = {} between neg = {}'.format(len(list(chain(*data_obj['pos']['within'].values()))),
    #     len(list(chain(*data_obj['neg']['within'].values()))),len(list(chain(*data_obj['pos']['between'].values()))),len(list(chain(*data_obj['neg']['between'].values())))
    # ))
    data_between = merge_edges(data_between_pos,data_between_neg)
    return data_within,data_between

#create 2 datasets: one for within edges, and the other for between edges
#using Hadamard because it was the best binary operator according to the article
def feature_nodes2edge(files_name,CD_model,data_name,has_two_learners,seed,has_communities,is_community_level):
    print("start converting nodes to edges")
    train_obj = read_gzip_object('{}_{}_{}_{}_train.gz'.format(data_name,CD_model,int(has_communities),seed,files_name[3]))
    test_obj = read_gzip_object('{}_{}_{}_{}_{}.gz'.format(data_name,CD_model,int(has_communities),seed,files_name[4]))
    G = read_gzip_object('{}_{}_{}_{}_{}.gz'.format(data_name,CD_model,int(has_communities),seed,files_name[0]))
    start = time()
    model = Features(G)
    train_within,train_between = create_feature_train_test(train_obj,model,'train')
    test_within,test_between = create_feature_train_test(test_obj,model,'test')
    # print('STOP EXECUTION')
    # exit()
    if not has_two_learners:
        scaler = MinMaxScaler(feature_range=(-1,1))
        scaler.fit(merge_edges(train_within[:,:-1],train_between[:,:-1]))
        train_within[:,:-1],train_between[:,:-1] = scaler.transform(train_within[:,:-1]), scaler.transform(train_between[:,:-1])
        test_within[:,:-1],test_between[:,:-1] =  scaler.transform(test_within[:,:-1]), scaler.transform(test_between[:,:-1])
    if has_two_learners:
        scaler_within = MinMaxScaler(feature_range=(-1,1))
        scaler_between = MinMaxScaler(feature_range=(-1,1))
        scaler_within.fit(train_within[:,:-1])
        scaler_between.fit(train_between[:,:-1])
        train_within[:,:-1],train_between[:,:-1] = scaler_within.transform(train_within[:,:-1]), scaler_between.transform(train_between[:,:-1])
        test_within[:,:-1],test_between[:,:-1] =  scaler_within.transform(test_within[:,:-1]), scaler_between.transform(test_between[:,:-1])

    # print('train within shape = {}\ttrain between shape = {}\ntest within shape = {}\t'
    #       'test between shape = {}'.format(train_within.shape,train_between.shape,test_within.shape,test_between.shape))
    print('It takes {} seconds to create edge embeddings'.format(time() - start))

    # save 4 datasets (within/between for train/test)
    train = {'within': train_within,'between':train_between}
    test = {'within': test_within,'between':test_between}
    return train,test
    # save_to_gzip(train,files_name[3]+'{}_{}_{}.gz'.format(seed,type,id))
    # save_to_gzip(test,files_name[4]+'{}_{}_{}.gz'.format(seed,type,id))


#save csv file where each row is comprised from:
    # node2vec parameters: p,q,d,r,l,k
    # model name (logistic regression, MLP, xgboost or Random Forest)
    # evaluation measures: acc?, F1, AP X3 (between,within,total)


## logistic regression and xgboost
def feature_link_prediction(model_name,has_two_learners,is_community_level,seed,
                            train_dict,test_dict,model_params_path ,output):
    # model_params = None
    # if model_params_path is not None:
    model_params = convert_file2obj(model_params_path)
    # if model_name == 'xgboost':
    #     model_params["max_depth"] = range(*model_params["max_depth"])
    #     model_params["min_child_weight"] = range(*model_params["min_child_weight"])
    model = Model(model_name,model_params,has_two_learners,is_community_level,seed)
    model.get_measures(train_dict,test_dict,output,model_name)







#has_communities:
# if 0 then need to use community detection algorithm to create communities
#CD_model:
# 0 if no need of CD algorithm, 1 for multilevel, 2 for infomap and 3 for label propagation


#is_community_level:
# 1 if community-level based node2vec should be created else (0) graph-level

# two types of prediction:
# type '1' - ignore community knowledge
# type '2' - differentiate between within and between community links
def exe(files_name,data_name,CD_model,model_name,has_two_learners,
        has_communities,is_community_level,seed,seed_trainings,model_params_path=None):

    output = []
    header = ["F1 within", "AP within", "Accuracy within", "AUC within", "F1 between", "AP between",
              "Accuracy between", "AUC between", "F1 total", "AP total", "Accuracy total", "AUC total", "params"]
    output.append(header)
    train_dict,test_dict = feature_nodes2edge(files_name,CD_model,data_name,has_two_learners,seed,has_communities,is_community_level)
    for seed_training in seed_trainings:
        feature_link_prediction(model_name,has_two_learners,is_community_level,seed_training,
                                train_dict,test_dict,model_params_path,output)
        save_res_to_csv(0,data_name,CD_model,has_communities,is_community_level,
                        has_two_learners,seed,seed_training,model_name,output)

        output = []
        output.append(header)
    return


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
    seed_trainings = input_params["seed_trainings"]
    is_synthetic = input_params["is_synthetic"]
    files_name = ['graph','community_dict','between_dict','train','test','embeddings','log']

    # community_dict = create_dateset(data_name,files_name,CD_model,has_communities=has_communities,is_synthetic=is_synthetic)
    # preprocess_G(data_name,files_name,CD_model,has_communities,community_dict,seed ,frac_within=0.5,frac_between=0.5)
    exe(files_name,data_name,CD_model,model_name,has_two_learners,has_communities,
                 is_community_level,seed,seed_trainings,model_params_path)


