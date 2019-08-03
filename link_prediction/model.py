import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from edge2vec import merge_edges
from sklearn.model_selection import StratifiedShuffleSplit,StratifiedKFold,GridSearchCV #,RandomizedSearchCV,,RandomizedSearchCV,train_test_split
from sklearn.metrics import average_precision_score,f1_score,confusion_matrix,roc_auc_score
# from sklearn.metrics import classification_report
# from file2obj import save_to_gzip,read_gzip_object,convert_file2obj,save_str
import itertools
import torch
import torch.nn as nn
# import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataset import TensorDataset
from torch.utils.data.dataloader import DataLoader
from torch.autograd import Variable
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from imblearn.over_sampling import RandomOverSampler
# import xgboost as xgb

from multiprocessing import cpu_count

dtype = torch.FloatTensor


'''
TODO:

* adjust the scorimg parameter in GridSearchCV,RandomizedSearchCV
  more info: http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
* complete train
* complete predict
* complete create_predicted_labeled_dict
* plot train-validation
'''



def get_f1_AP_acc_AUC_scores(y,y_pred,y_pred_proba):
    f1 = f1_score(y,y_pred)
    ap = average_precision_score(y,y_pred_proba)
    acc = np.mean(y == y_pred)
    auc = roc_auc_score(y,y_pred_proba)
    return f1,ap,acc,auc

def create_CM_plot(y_test,y_pred,path,title="Confusion Matrix",classes = ['link','non-link']):
    # print('accuracy = {}'.format(np.mean(y_test == y_pred)))
    # print('fraction of positive total = {}'.format(np.sum(y_pred == 1) / len(y_pred)))
    cm = confusion_matrix(y_test, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    np.set_printoptions(precision=2)
    # print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f'
    thresh = cm.max() / 2.0
    # print('threshold equals to {}'.format(thresh))
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(path)
    plt.close()

def fix_predictions(preds):
    preds[preds != 0] = 1

# if we want to re-initialize all our parameters
def reset(m):
    if hasattr(m, 'reset_parameters'):
        m.reset_parameters()
    # if hasattr(m, 'reset_parameters') and hasattr(m, 'requires_grad'):
    #     if m['requires_grad']:
    #         m.reset_parameters()
    # else:
    #     print('no reset was done on this layer')
    # else:
        # print("fail to reset")

def get_loader(X_numpy,y_numpy,batch_size = 128):
    X, y = torch.from_numpy(X_numpy), torch.from_numpy(y_numpy)
    dataset = TensorDataset(X,y)
    loader = DataLoader(dataset,batch_size = batch_size,shuffle=True)
    return loader

def print_architecture(architecture):
    N = len(architecture) - 1
    h = '('
    for i,W in enumerate(architecture):
        if i == N:
            h += '{})'.format(W[0])
        else:
            h += '{},'.format(W[0])
    return h

def get_modules(h_sizes,add_output=True):
    modules = []
    for in_out in h_sizes[:-1]:
        modules.append(nn.Linear(*in_out))
        modules.append(nn.ReLU(inplace=True))
    if add_output:
        modules.append(nn.Linear(*h_sizes[-1]))
    return modules



class Model(object):
    #params list of tuples (first elemet:name of a layer, the rest: parameters)
    def __init__(self,model_params,to_diff_links,is_multiclass,method,seed=1,frac_train = 0.8):
        self.model_params = model_params
        # self.model_dict = {'Logistic Regression':LogisticRegressionCV}#add xgboost
        self.to_diff_links = to_diff_links
        self.classification_dict = {(0,0):"standard",(0,1): "multiclass",(1,0):"embedded"}
        self.is_multiclass = is_multiclass
        self.num_epochs = 21 #instead of 21
        self.n_splits = 1 #instead of 5
        self.step = 5 #instead of 10
        self.method = method
        self.seed = seed
        self.frac_train = frac_train
        np.random.seed(seed)
        random.seed(seed)


    def wiggly_train_plot(self,train,filename):
        N = len(train)
        it_per_epoch = int(N / self.num_epochs)
        x = [ i/it_per_epoch for i in range(N)]
        fig,ax = plt.subplots()
        ax.set_xticks(list(i for i in range(0,self.num_epochs+1,self.step)))
        plt.plot(x,train,color='b')
        plt.title("Training loss")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.savefig(filename)
        plt.close()


    def train_val_plot(self,train,val,filename):
        x = [ i for i in range(self.num_epochs)]
        fig,ax = plt.subplots()
        ax.set_xticks(list(i for i in range(0,self.num_epochs,self.step)))
        plt.plot(x,train,color='b',label = 'train')
        plt.plot(x,val,color='r',label = 'val')
        plt.title("Training vs Valiadtion loss")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.legend(loc = 'upper right')
        plt.savefig(filename)
        plt.close()

    def lr_plot(self,lrs,labels,filename):
        N = len(lrs)
        colors = cm.Pastel1(list(range(N)))
        x = [ i for i in range(self.num_epochs)]
        fig,ax = plt.subplots()
        ax.set_xticks(list(i for i in range(0,self.num_epochs,self.step)))
        for i,lr in enumerate(lrs):
            plt.plot(x,lr,label = "lr = {}".format(labels[i]),color=colors[i])
        plt.title("Learning rates loss")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.legend(loc = 'upper right')
        plt.savefig(filename)
        plt.close()

    # def mlp_train(self,model,loss,optimizer,loader_train,print_every = 500):
    #     # model.apply(reset)
    #     model.train()
    #     for t,(train, target) in enumerate(loader_train):
    #
    #         tr_x, tr_y = Variable(train.float()), Variable(target.long())
    #         # if cuda:
    #         #     tr_x, tr_y = tr_x.cuda(), tr_y.cuda()
    #         # Reset gradient
    #         optimizer.zero_grad()
    #
    #         # Forward pass
    #         fx = model(tr_x)
    #         output = loss(fx, tr_y) #loss for this batch
    #         if (t + 1) % print_every == 0:
    #              print('t = %d, loss = %.4f' % (t + 1, output.item()))
    #         # Backward
    #         output.backward()
    #         # Update parameters based on backprop
    #         optimizer.step()



    def mlp_train_val(self,model, loss, optimizer,loader_train,loader_val,print_every = 100):
        train_loss=[]
        val_loss = []
        lr_loss = []
        diverge = 1.0 * 3 #(initial (expected) loss times k (k=3/4))
        model.train()
        for epoch in range(self.num_epochs):

            if (epoch + 1) % 5 == 0:
                print('epoch number {}'.format(epoch+1))
            for t,(train, target_train) in enumerate(loader_train):
                tr_x, tr_y = Variable(train.float()), Variable(target_train.long())
                # if cuda:
                #     tr_x, tr_y = tr_x.cuda(), tr_y.cuda()
                # Reset gradient
                optimizer.zero_grad()

                # Forward pass
                fx = model(tr_x)
                output = loss(fx, tr_y) #loss for this batch
                # if (t + 1) % (print_every // 2) == 0:
                    # if (t + 1) % print_every == 0:
                    #     print('t = %d, loss = %.4f' % (t + 1,output.item())) #change item instead of output2.data[0].cpu().numpy()
                train_loss_val = output.item() #change item instead of numpy()
                train_loss.append(train_loss_val)# / (print_every - 1))


                # Backward
                output.backward()

                # Update parameters based on backprop
                optimizer.step()

            model.eval()
            avg_val_loss = 0.0
            val_size = 0
            for val,traget_val in loader_val:
                vl_x, vl_y = Variable(val.float()), Variable(traget_val.long())
                batch_size = vl_x.shape[0]
                val_size += batch_size
                fx2 = model(vl_x)
                output2 = loss(fx2, vl_y) #loss for this batch
                avg_val_loss += output2.item() * batch_size #change item instead of output2.data[0].cpu().numpy()
            avg_val_loss /= val_size
            val_loss.append(avg_val_loss)
            avg_train_loss = 0.0
            train_size = 0
            for train, target_train in loader_train:
                tr_x, tr_y = Variable(train.float()), Variable(target_train.long())
                batch_size = tr_x.shape[0]
                train_size += batch_size
                fx1 = model(tr_x)
                output1 = loss(fx1, tr_y)
                avg_train_loss += output1.item() * batch_size

            avg_train_loss /= train_size
            lr_loss.append(avg_train_loss)

            #if has diverged
            if avg_train_loss > diverge:
                print('skip architecture (diverged)')
                return [],[],[]

            model.train()


        return train_loss,val_loss,lr_loss

    def mlp_predict(self,model,X_test,y_test,type):
        model.eval()
        preds = None
        proba = None
        X_test, y_test = torch.from_numpy(X_test), torch.from_numpy(y_test)
        test_within = TensorDataset(X_test,y_test)
        loader_test = DataLoader(test_within,batch_size = X_test.shape[0])
        for x, _ in loader_test:
            x_var = Variable(x.type(dtype))
            scores = model(x_var)
            # if error erase cpu() (because already in cpu state)
            # _, batch_preds = scores.data.cpu().max(1)
            _, preds = scores.data.cpu().max(1)
            preds = preds.numpy()
            # preds.append(batch_preds.numpy())
            Softmax = torch.nn.Softmax(dim=1)
            proba = Softmax(scores).data.cpu().numpy()
            # probas.append(proba.numpy())
        proba = proba[:,1:].sum(axis=1)
        # print("probas = {}\npreds = {}".format(probas,preds))
        # print("preds 'is link' ({}) frac = {}".format(type,np.sum(preds == 1) / len(preds)))
        return preds,proba#np.vstack(preds),np.vstack(probas)

    def CV(self,model,optimizer,loss,train):
        sfk = StratifiedKFold(n_splits=self.n_splits)
        train_mat = []
        val_mat = []
        lr_mat = []
        for train_index, val_index in sfk.split(train[:,:-1],train[:,-1]):
            model.apply(reset)
            loader_train = get_loader(train[train_index,:-1],train[train_index,-1])
            loader_val = get_loader(train[val_index,:-1],train[val_index,-1],train[val_index,:].shape[0])
            train_loss,val_loss,lr_loss = self.mlp_train_val(model, loss, optimizer,loader_train,loader_val,print_every = 100)
            if len(train_loss) + len(val_loss) + len(lr_loss) == 0:
                return [],[],[]
            train_mat.append(train_loss)
            val_mat.append(val_loss)
            lr_mat.append(lr_loss)
        train_mat = np.array(train_mat).mean(axis = 0)
        val_mat = np.array(val_mat).mean(axis = 0)
        lr_mat = np.array(lr_mat).mean(axis = 0)
        return train_mat,val_mat,lr_mat

    def validation(self,model,optimizer,loss,train):
        sss = StratifiedShuffleSplit(n_splits=self.n_splits,test_size=1-self.frac_train,random_state=self.seed)
        # train_arr = []
        # val_arr = []
        # lr_arr = []
        for train_index, val_index in sss.split(train[:,:-1],train[:,-1]):
            model.apply(reset)
            loader_train = get_loader(train[train_index,:-1],train[train_index,-1])
            loader_val = get_loader(train[val_index,:-1],train[val_index,-1],train[val_index,:].shape[0])
            train_loss,val_loss,lr_loss = self.mlp_train_val(model, loss, optimizer,loader_train,loader_val,print_every = 100)
            if len(train_loss) + len(val_loss) + len(lr_loss) == 0:
                return [],[],[]
            else:
                train_arr = np.array(train_loss)
                val_arr = np.array(val_loss)
                lr_arr = np.array(lr_loss)
                return train_arr,val_arr,lr_arr

    def get_mlp_model(self,classification_type,architecture):
        if classification_type == "standard" or classification_type == "multiclass":
            return nn.Sequential(*get_modules(architecture)).type(dtype)
        else: #embedded
            model = nn.Sequential(*get_modules(architecture)).type(dtype)
            state_dict = torch.load('best_edge_model.pth')

            return model


    def get_MLP_measures(self,train_dict,test_dict,output,params_str,verbose = False):
        classification_type = self.classification_dict[(self.to_diff_links,self.is_multiclass)]
        optimizer_names = ["adam","SGD"]
        train,test_within,test_between = self.get_train_test(train_dict,test_dict)
        params = self.model_params[classification_type]
        architectures = params["architectures"]

        ######### TODO ##########
        #1. Adjust dataset if to_diff_links or is multi_class
        #  (else treat it as regular)
        #2. check if training is OK
        classes = ["non-link","link"]
        if self.is_multiclass:
            classes = ["non-link","within-link","between-link"]
        for architecture in architectures:
            model = self.get_mlp_model(classification_type,architecture)
            for optimizer_name in optimizer_names:
                #Cross Entropy Loss
                loss = nn.CrossEntropyLoss().type(dtype)
                if optimizer_name == "adam" and optimizer_name in params.keys():
                    lrs = params[optimizer_name]["lr"]
                    lr_means = []
                    rel_lrs = []
                    for lr in lrs:
                        info = "total_params_arc={}_optim={}_lr={}".format(print_architecture(architecture),optimizer_name,lr)
                        optimizer = optim.Adam(model.parameters(),lr = lr)
                        train_mean,val_mean,lr_mean = self.validation(model,optimizer,loss,train)
                        if len(train_mean) + len(val_mean) + len(lr_mean) == 0:
                            print('{} was skipped'.format(info))
                            continue

                        y_within_pred,y_within_pred_proba = self.mlp_predict(model,test_within[:,:-1],test_within[:,-1],'within')
                        y_between_pred,y_between_pred_proba = self.mlp_predict(model,test_between[:,:-1],test_between[:,-1],'between')

                        frac = float(np.sum(y_within_pred == 1)) / len(y_within_pred)
                        if  frac== 0.0 or frac == 1.0:
                            print('there was no learning, and therefore {} was skipped'.format(info))
                            continue

                        create_CM_plot(np.hstack((test_within[:,-1],test_between[:,-1])),np.hstack((y_within_pred,y_between_pred)),
                                       "plots_nn/{}_arc={}_optim={}_lr={}.png".format(self.method, print_architecture(architecture),
                                        optimizer_name,lr),classes=classes)
                        binary_test_within = np.array(test_within[:,-1])
                        binary_test_between= np.array(test_between[:,-1])
                        if self.is_multiclass:
                            y_within_pred = fix_predictions(y_within_pred)
                            binary_test_within = fix_predictions(binary_test_within)
                            y_between_pred = fix_predictions(y_between_pred)
                            binary_test_between = fix_predictions(binary_test_between)
                        self.wiggly_train_plot(train_mean,'plots_nn/wiggly_{}.png'.format(info))
                        self.train_val_plot(lr_mean,val_mean,'plots_nn/train_val_{}.png'.format(info))
                        lr_means.append(lr_mean)
                        rel_lrs.append(lr)
                        f1_within,ap_within,acc_within,auc_within = get_f1_AP_acc_AUC_scores(binary_test_within,y_within_pred,y_within_pred_proba)

                        f1_between,ap_between,acc_between,auc_between = get_f1_AP_acc_AUC_scores(binary_test_between,y_between_pred,y_between_pred_proba)
                        f1_total,ap_total,acc_total,auc_total = get_f1_AP_acc_AUC_scores(np.hstack((binary_test_within,binary_test_between)),np.hstack((y_within_pred,y_between_pred)),
                                                                            np.hstack((y_within_pred_proba,y_between_pred_proba)))
                        output.append(params_str+[f1_within,ap_within,acc_within,auc_within,f1_between,ap_between,acc_between,auc_between,
                                                  f1_total,ap_total,acc_total,auc_total,"params:\narc={},optim={},lr={}".format(
                                                    print_architecture(architecture),optimizer_name,lr)])
                    if len(lr_means) > 0:
                        self.lr_plot(lr_means,rel_lrs,'plots_nn/LR_plot_{}.png'.format("params_arc={}_optim={}".format(print_architecture(architecture),optimizer_name)))
                elif optimizer_name == "SGD" and optimizer_name in params.keys():
                    lrs = params[optimizer_name]["lr"]
                    momentums = params[optimizer_name]["momentum"]
                    weights_decay = params[optimizer_name]["weight_decay"]
                    for momentum in momentums:
                        for weight_decay in weights_decay:
                            lr_means = []
                            rel_lrs = []
                            for lr in lrs:
                                info = "total_params_arc={}_optim={}_lr={}_rho={}_l2={}".format(print_architecture(architecture),optimizer_name,lr,momentum,weight_decay)
                                optimizer = optim.SGD(model.parameters(),lr=lr,momentum=momentum,
                                                      weight_decay=weight_decay,nesterov=True)
                                train_mean,val_mean,lr_mean = self.validation(model,optimizer,loss,train)
                                if len(train_mean) + len(val_mean) + len(lr_mean) == 0:
                                    print('{} was skipped'.format(info))
                                    continue

                                y_within_pred,y_within_pred_proba = self.mlp_predict(model,test_within[:,:-1],test_within[:,-1],'within')
                                y_between_pred,y_between_pred_proba = self.mlp_predict(model,test_between[:,:-1],test_between[:,-1],'between')

                                create_CM_plot(np.hstack((test_within[:,-1],test_between[:,-1])),np.hstack((y_within_pred,y_between_pred)),
                                       "plots_nn/{}_arc={},optim={},lr={}momentum={},weight_decay={}.png".format(self.method, print_architecture(architecture),
                                        optimizer_name,lr,momentum,weight_decay),classes=classes)
                                binary_test_within = np.array(test_within[:,-1])
                                binary_test_between= np.array(test_between[:,-1])
                                if self.is_multiclass:
                                    y_within_pred = fix_predictions(y_within_pred)
                                    binary_test_within = fix_predictions(binary_test_within)
                                    y_between_pred = fix_predictions(y_between_pred)
                                    binary_test_between = fix_predictions(binary_test_between)

                                frac = float(np.sum(y_within_pred == 1)) / len(y_within_pred)
                                if  frac== 0.0 or frac == 1.0:
                                    print('there was no learning, and therefore {} was skipped'.format(info))
                                    continue
                                self.wiggly_train_plot(train_mean,'plots_nn/wiggly_{}.png'.format(info))
                                self.train_val_plot(lr_mean,val_mean,'plots_nn/train_val_{}.png'.format(info))
                                #plot wiggle training and validation loss
                                lr_means.append(lr_mean)
                                rel_lrs.append(lr)
                                #train_loader = get_loader(train[:,:-1],train[:,-1])
                                #self.mlp_train(model, loss, optimizer,train_loader,print_every = 500)


                                f1_within,ap_within,acc_within,auc_within = get_f1_AP_acc_AUC_scores(binary_test_within,y_within_pred,y_within_pred_proba)
                                # if self.is_community_level:
                                #     output.append(n2v_params+[f1_within,ap_within,acc_within,acc_total,
                                #                               "total params:\narc={},optim={},lr={}\nmomentum={},weight_decay={}".format(
                                #                             print_architecture(architecture),optimizer_name,lr,momentum,weight_decay)])
                                f1_between,ap_between,acc_between,auc_between = get_f1_AP_acc_AUC_scores(binary_test_between,y_between_pred,y_between_pred_proba)
                                f1_total,ap_total,acc_total,auc_total = get_f1_AP_acc_AUC_scores(np.hstack((binary_test_within,binary_test_between)),np.hstack((y_within_pred,y_between_pred)),
                                                                                    np.hstack((y_within_pred_proba,y_between_pred_proba)))
                                output.append(params_str+[f1_within,ap_within,acc_within,auc_within,f1_between,ap_between,acc_between,auc_between,f1_total,ap_total,
                                                          acc_total,auc_total,"params:\narc={},optim={},lr={}\nmomentum={},weight_decay={}".format(
                                                            print_architecture(architecture),optimizer_name,lr,momentum,weight_decay)])
                            if len(lr_means) > 0:
                                self.lr_plot(lr_means,rel_lrs,'plots_nn/LR_plot_{}.png'.format("params_arc={}_optim={}_rho={}_l2={}".format(print_architecture(architecture),optimizer_name,momentum,weight_decay)))

    def oversampler(self,x1,y1,x2,y2):
        ros = RandomOverSampler(random_state=self.random_state)
        x1[:,-1] = y1
        x2[:,-1] = y2
        Xy = np.vstack(x1,x2)
        return ros.fit_resample(Xy[:,:-1],Xy[:,-1])

    def downsampler(self,X,size):
        N = X.shape[0]
        idxes = np.random.choice(range(N),size,replace=False)
        return np.stack([X[idx,:] for idx in idxes])

    def set_y(self,X_arr,y_arr):
        for i,X in enumerate(X_arr):
            X[:,-1] = y_arr[i]

    def get_train_test(self,train_dict,test_dict):
        train_within_pos,train_within_neg = train_dict['within']['pos'],train_dict['within']['neg']
        train_between_pos,train_between_neg = train_dict['between']['pos'],train_dict['between']['neg']
        test_within_pos,test_within_neg = test_dict['within']['pos'],test_dict['within']['neg']
        test_between_pos,test_between_neg = test_dict['between']['pos'],test_dict['between']['neg']

        if self.to_diff_links: #embedded classification
            N = int(train_between_pos.shape[0] * 2 *self.frac_train)
            between_pos = self.downsampler(train_between_pos,(N // 2))
            range_idx = range(len(train_within_pos))
            if len(train_within_pos) > (N // 2):
                range_idx = range(N // 2)
            X_b_w_pos, y_b_w_pos = self.oversampler(train_within_pos[range_idx,:],0,between_pos,1)
            y_b_w_pos[:] = 1
            between_neg = self.downsampler(train_between_neg,(N // 2))
            range_idx = range(len(train_within_neg))
            if len(train_within_neg) > (N // 2):
                range_idx = range(N // 2)
            X_b_w_neg, y_b_w_neg = self.oversampler(train_within_neg[range_idx,:],0,between_neg,1)
            y_b_w_neg[:] = 0
            train = np.stack(np.hstack(X_b_w_pos,y_b_w_pos),np.hstack(X_b_w_neg,y_b_w_neg))
            self.set_y([test_between_pos,test_within_pos,test_between_neg,test_within_neg],[1,1,0,0])
            test_within = merge_edges(test_within_pos,test_within_neg)
            test_between = merge_edges(test_between_pos, test_between_neg)
            return train,test_within,test_between
        else:
            if self.is_multiclass: #multiclass classification
                n_between = train_between_pos.shape[0]
                X_b_w_pos, y_b_w_pos = self.oversampler(train_within_pos,1,train_between_pos,2)
                between_neg = self.downsampler(train_between_neg,(n_between // 2))
                range_idx = range(len(train_within_neg))
                if len(train_within_neg) > (n_between // 2):
                    range_idx = range(n_between // 2)
                X_b_w_neg, y_b_w_neg = self.oversampler(train_within_neg[range_idx,:],0,between_neg,1)
                y_b_w_neg[:] = 0
                train = np.stack(np.hstack(X_b_w_pos,y_b_w_pos),np.hstack(X_b_w_neg,y_b_w_neg))
                self.set_y([test_between_pos,test_within_pos,test_between_neg,test_within_neg],[2,1,0,0])
                # test_between_pos[:,-1] = 1
                # test_within_pos[:,-1] = 0
                # test_between_neg[:,-1] = 2
                # test_within_neg[:,-1] = 2
                test_within = merge_edges(test_within_pos,test_within_neg)
                test_between = merge_edges(test_between_pos, test_between_neg)
                return train,test_within,test_between
            else: #standard binary classification
                train_within = merge_edges(train_within_pos,train_within_neg)
                train_between = merge_edges(train_between_pos, train_between_neg)
                train = merge_edges(train_within,train_between)
                test_within = merge_edges(test_within_pos,test_within_neg)
                test_between = merge_edges(test_between_pos, test_between_neg)
                return train,test_within,test_between

    def get_LR_model(self,c):
        num_cores = cpu_count()
        model = LogisticRegression(C=c,random_state=1) #,random_state = self.model_params['random_state']
        return model

    def get_xgboost_model(self,params):
        max_depth,lr = params
        model = XGBClassifier(max_depth=max_depth,learning_rate=lr,n_jobs=cpu_count()) #random_state=self.seed, seed=self.seed)
        return model


    def model_predict(self,model,X_test):
        y_pred = model.predict(X_test)
        # print('fraction of positive preds: {}'.format(np.sum(y_pred == 1) / len(y_pred)))
        # print('pred classes: {}'.format(np.unique(y_pred)))
        y_pred_proba = model.predict_proba(X_test)[:,-1]
        return y_pred,y_pred_proba


    def get_measures(self,train_dict,test_dict,output,model_name,str_params=None,verbose = False):
        train,test_within,test_between = self.get_train_test(train_dict,test_dict)

        get_model = None
        if model_name == 'Logistic Regression':
            params = self.model_params["Cs"]
            get_model = self.get_LR_model
        else:
            max_depth = self.model_params["max_depth"]
            lr = self.model_params["learning_rate"]
            params = [(max_depth,lr)]
            get_model = self.get_xgboost_model

        sss = StratifiedShuffleSplit(n_splits=self.n_splits,test_size=1-self.frac_train,random_state=self.seed)
        best_params = None
        for train_index, val_index in sss.split(train[:,:-1],train[:,-1]):
            for param in params:
                model = get_model(param)
                model.fit(train[train_index, :-1], train[train_index, -1])
                y_within_pred,y_within_pred_proba = self.model_predict(model,test_within[:,:-1])
                y_within = test_within[:,-1]
                f1_within,ap_within,acc_within,auc_within = get_f1_AP_acc_AUC_scores(y_within,y_within_pred,y_within_pred_proba)

                y_between_pred,y_between_pred_proba = self.model_predict(model,test_between[:,:-1])
                y_between = test_between[:,-1]
                f1_between,ap_between,acc_between,auc_between = get_f1_AP_acc_AUC_scores(y_between,y_between_pred,y_between_pred_proba)
                f1_total,ap_total,acc_total,auc_total = get_f1_AP_acc_AUC_scores(np.hstack((y_within,y_between)),np.hstack((y_within_pred,y_between_pred)),
                                                 np.hstack((y_within_pred_proba,y_between_pred_proba)))
                if  model_name == 'Logistic Regression':
                    best_params = param
                else:
                    best_params = model.get_xgb_params()

                if verbose:
                    print('Model total params:\n{}'.format(best_params))
                    #print('Model best c is c = {}'.format(model.C_[0]))
                    print('y_between_pred == 1: {}\ny_between_pred == 0: {}'.format(np.sum(y_between_pred == 1),np.sum(y_between_pred == 0)))

                # if str_params is None:
                #     output.append([f1_within,ap_within,acc_within,auc_within,f1_between,ap_between,acc_between,auc_between,f1_total,
                #     ap_total,acc_total,auc_total,"total model params = {}".format(best_params)])
                # else:
                output.append(str_params+[f1_within,ap_within,acc_within,auc_within,f1_between,ap_between,acc_between,auc_between,
                                          f1_total,ap_total,acc_total,auc_total,"total model params = {}".format(best_params)])

        return
