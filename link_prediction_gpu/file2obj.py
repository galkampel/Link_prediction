import pickle
import json
import gzip
import networkx as nx
import csv
##note-  string are denoted by "" and not by ''
def convert_file2obj(path):
    fin = open(path,'rb')
    model= json.load(fin)
    fin.close()
    return model

#### compressed object (ends with .gz)
def save_to_gzip(obj,path):
    fout = gzip.GzipFile(path,'wb')
    fout.write(pickle.dumps(obj))
    fout.close()

def read_gzip_object(path):
    fin = gzip.GzipFile(path,'rb')
    obj = pickle.loads(fin.read())
    fin.close()
    return obj
## read graph from pickle
def read_graph_pickle(path):
    G = nx.read_gpickle(path)
    return G
def save_graph_gpickle(G,path):
    nx.write_gpickle(G,path)

#### more expensive than gzip
def save_to_pickle(obj,path):
        fout = open(path, 'wb')
        pickle.dump(obj,fout)
        fout.close()

def read_pickled_object(path):
    fin = open(path, 'rb')
    vectors = pickle.load(fin)
    fin.close()
    return vectors


def read_csv_params(path):
    params = {}
    file_reader = csv.reader(open(path, newline=''), delimiter=',')#, quotechar='|')
    keys = next(file_reader,None)
    N = len(keys)
    for count,row in enumerate(file_reader):
        params[count] = {keys[i]:float(row[i]) if keys[i] == 'p' or keys[i] == 'q' else int(row[i]) for i in range(N)}
    return params


def save_res_to_csv(count,data_name,CD_model,has_communities,to_diff,
                    is_multiclass,seed,seed_training,model_name,output):

    with open('{}_{}_{}{}{}{}{}_trnSeed={}_Model={}.csv'.format(count,data_name,CD_model,int(has_communities),int(to_diff),int(is_multiclass),
                                                      seed,seed_training,model_name.replace(" ","_")),mode='w',newline='') as fout:
        writer = csv.writer(fout)
        writer.writerows(output)

def save_to_csv(output,path='best_all_params.csv'):
    with open(path,'w',newline='') as fout:
        writer = csv.writer(fout)
        writer.writerows(output)



def save_labels(G,path):
    fout = open(path, 'w')
    nodes = G.nodes(data=False)
    for node in nodes:
        fout.write("{} {}\n".format(node,G.nodes[node]['label']))
    fout.close()

def save_str(obj,path):
    fout = open(path,'wb')
    fout.write(obj.encode('utf-8'))
    fout.close()


