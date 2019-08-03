from allennlp.modules.elmo import Elmo, batch_to_ids
import numpy as np
# from file2obj import save_to_gzip
# from multiprocessing import cpu_count
# from joblib import Parallel,delayed
from time import  time


class ElmoEmbeddings:
    def __init__(self,scalar_mix_parameters,num_output_representations=1,dropout= 0.5,requires_grad=False,embeddings_dim = 1024):
        self.num_output_representations = num_output_representations
        self.dropout = dropout
        self.requires_grad = requires_grad
        self.embeddings_dim = 1024
        self.scalar_mix_parameters = scalar_mix_parameters

    def set_ELMO_embeddings(self,sentences):
        options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
        weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

        elmo = Elmo(options_file, weight_file, num_output_representations=self.num_output_representations,
                    dropout=self.dropout,requires_grad=self.requires_grad,scalar_mix_parameters=self.scalar_mix_parameters)
        character_ids = batch_to_ids(sentences)
        embeddings = elmo(character_ids)
        emb=embeddings['elmo_representations'] #a list of the embeddings of output_representations with dim (batch_size (N), timesteps (l), embedding_dim (1024))
        self.emb = emb
        return

    def get_vectors(self,nodes2idx):
        emb = self.emb
        vectors = {}
        for node in nodes2idx.keys():
            w_i = np.zeros(self.embeddings_dim)
            N = len(nodes2idx[node])
            for s,w in nodes2idx[node]:
                w_i += emb[0][s][w].detach().numpy()  #the word embeddings of w^{th} word in the s^{th} sentence
            w_i +=  w_i / N
            vectors[node] = w_i
        return vectors

class ComE:
    def __init__(self):
        2

class NMF:
    def __init__(self):
        3