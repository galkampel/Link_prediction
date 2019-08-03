
import numpy as np
import torch
import torch.nn as nn
# import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from imblearn.over_sampling import RandomOverSampler
from file2obj import convert_file2obj
# import matplotlib
import matplotlib.pyplot as plt
from model_gpu import reset,get_loader,get_modules
from sklearn.manifold import TSNE


normalize = lambda A: (A.T / A.sum(axis=1) ).T
project = lambda X:  TSNE(n_components=2).fit_transform(X)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.cuda.FloatTensor


#TODO
#* save best model
#*

class Net(nn.Module):
    def __init__(self, hs):

        self.to_embed = False
        super(Net, self).__init__()
        self.model = nn.Sequential(*get_modules(hs,False)).type(dtype)
        self.output = nn.Linear(*hs[-1])

    def set_to_embed(self,to_embed):
        self.to_embed = to_embed

    def forward(self, x):
        embedded = self.model(x)
        if self.to_embed:
            return embedded
        else:
            y_pred = self.output(embedded)
            return y_pred

class Splitter(object):
    def __init__(self,method,num_epochs = 101,step = 10,seed=0, save_best_model=True):
        self.method = method
        self.diverge = 1.0 * 3 #(initial (expected) loss times k (k=3/4))
        self.num_epochs = num_epochs #instead of 21
        self.step = step
        self.seed = seed
        self.random_state = seed
        self.optimizers_name = ["SGD","adam"]
        self.save_best_model = save_best_model
        np.random.seed(seed)


    def link_split(self,links,frac):
        n = int(links.shape[0] * frac)
        mask = np.array( [False]*links.shape[0])
        idxes = np.random.choice(range(links.shape[0]),n,replace=False)
        mask[idxes] = True
        train = links[idxes]
        test = links[~idxes]
        return train,test


    def get_balanced_within_between_data(self,w_links,b_links):
        X = np.vstack((w_links[:,:-1],b_links[:,:-1]))
        y = np.vstack((np.ones((len(w_links),1)),np.zeros((len(b_links),1)))).reshape((-1,))
        ros = RandomOverSampler(random_state=self.random_state)
        X_res, y_res = ros.fit_resample(X, y)
        # print('X shape = {}\ty shape = {}\nX_res shape = {}\ty_res shape = {}'.format(X.shape,y.shape,X_res.shape,y_res.shape))
        data = np.hstack((X_res,y_res.reshape((-1,1))))
        return data[:,:-1], data[:,-1]

    def get_optim(self,model,optimizer_name,params):
        if optimizer_name == "SGD":
            return optim.SGD(model.parameters(),lr=params[0],momentum=params[1],
                        weight_decay=params[2],nesterov=True),\
                   "optim=SGD_lr={}_momentum={}_weight_decay={}".format(params[0],params[1],params[2])
        else:
            return optim.Adam(model.parameters(),lr = params),"optim=adam_lr={}".format(params)

    def readjust_params(self,params_dict,optimizer_name):
        if optimizer_name == "SGD":
            return [(lr,momentum,w_decay) for lr in  params_dict[optimizer_name]["lr"]
                    for momentum in params_dict[optimizer_name]["momentum"]
                    for w_decay in params_dict[optimizer_name]["weight_decay"]]
        else:
            return [lr for lr in  params_dict[optimizer_name]["lr"]]



    def train_val_plot(self,train,val,filename):
            x = [ i for i in range(self.num_epochs)]
            fig,ax = plt.subplots()
            ax.set_xticks(list(i for i in range(0,self.num_epochs,self.step)))
            plt.plot(x, train, color='b', label='train')
            plt.plot(x, val, color='r', label='val')
            plt.title("Training vs Validation loss")
            plt.ylabel("loss")
            plt.xlabel("epoch")
            plt.legend(loc = 'upper right')
            plt.savefig(filename)
            plt.close()

    def TSNE_plot(self,links,num_b,title,colors = ('skyblue', 'orange')):
        plt.figure()
        plt.scatter(links[:num_b,0],links[:num_b,1],color= colors[0],label = 'between link')
        plt.scatter(links[num_b:,0],links[num_b:,1],color= colors[1],label = 'within link')
        plt.title('TSNE {}'.format(title))
        plt.legend(loc = 'upper right')
        plt.savefig('plots_edge2vec/{}_{}_{}.png'.format(self.method,self.optim_str,title.split()[0]))
        plt.close()

    def visuallize(self,best_model,b_links,w_links,n=500):
        best_model.eval()
        new_b_links = normalize(best_model(torch.from_numpy(b_links).float().to(device)).data.cpu().numpy())
        new_w_links = normalize(best_model(torch.from_numpy(w_links).float().to(device)).data.cpu().numpy())
        # new_b_links = normalize(best_model(torch.from_numpy(b_links).cuda()).data.cpu().numpy())
        # new_w_links = normalize(best_model(torch.from_numpy(w_links).cuda()).data.cpu().numpy())
        num_b = b_links.shape[0]
        num_w = w_links.shape[0]
        idx_b = np.random.choice(range(num_b),n,replace=False)
        idx_w = np.random.choice(range(num_b,num_b+num_w),n,replace=False)
        print('new_b_links:\ttype = {}\tshape = {}\nnew_w_links:\ttype = {}\tshape = {}'.format(
            type(new_b_links),new_b_links.shape,type(new_w_links),new_w_links.shape))
        low_links = project(np.vstack((b_links,w_links))[np.concatenate((idx_b,idx_w)),:])
        new_low_links = project(np.vstack((new_b_links,new_w_links))[np.concatenate((idx_b,idx_w)),:])

        self.TSNE_plot(new_low_links,n,'before embbeding')
        self.TSNE_plot(low_links,n,'after embbeding')


    def calc_epoch_loss(self,model,loss,loader):
        avg_loss = 0.0
        size = 0
        for X, y in loader:
            X_var, y_var = Variable(X.float().cuda()), Variable(y.long().cuda())
            batch_size = X_var.shape[0]
            size += batch_size
            fx = model(X_var)
            output = loss(fx, y_var)  # loss for this batch
            avg_loss += output.data.cpu().numpy() * batch_size  # output.item()   change item instead of output2.data[0].cpu().numpy()
        avg_loss /= size
        return avg_loss

    def set_within_between_sep(self,w_links,b_links,params_path='split_wb.json',frac_between = 0.8,frac_within=0.9): #frac_between = 0.75,frac_within=0.8

        b_train,b_test = self.link_split(b_links,frac_between)
        w_train,w_test = self.link_split(w_links,frac_within)
        X,y = self.get_balanced_within_between_data(w_train,b_train)
        params_dict = convert_file2obj(params_path)
        params_dict["SGD"] = self.readjust_params(params_dict,"SGD")
        params_dict["adam"] = self.readjust_params(params_dict,"adam")
        model = Net(params_dict["architecture"]).type(dtype)
        loader_train = get_loader(X,y)
        x_test = np.vstack((w_test[:,:-1],b_test[:,:-1]))
        y_test = np.vstack((w_test[:,-1].reshape((-1,1)),b_test[:,-1].reshape((-1,1)))).reshape((-1,))
        loader_val = get_loader(x_test,y_test)
        # X_test,y_test = self.get_balanced_within_between_data(w_test,b_test,to_balance=False)
        N = X.shape[0]
        best_loss = np.inf
        for optimizer_name in self.optimizers_name:
            for params in params_dict[optimizer_name]:
                model.apply(reset)
                optimizer,optim_str = self.get_optim(model,optimizer_name,params)
                loss = nn.CrossEntropyLoss().type(dtype)
                train_loss_lst = []
                val_loss_lst = []
                has_diverged = False
                for epoch in range(self.num_epochs):
                    model.train()
                    train_loss = 0.0
                    if (epoch + 1) % 5 == 0:
                        print('epoch number {}'.format(epoch+1))
                    for t,(train, target_train) in enumerate(loader_train):
                        tr_x, tr_y = Variable(train.float().cuda()), Variable(target_train.long().cuda())
                        # Reset gradient
                        optimizer.zero_grad()

                        # Forward pass
                        fx = model(tr_x)
                        output = loss(fx, tr_y) #loss for this batch
                        batch_size = tr_x.shape[0]
                        # train_loss += output.data.numpy() * batch_size
                        train_loss += output.data.cpu().numpy() * batch_size #change item instead of numpy()

                        # Backward
                        output.backward()
                        # Update parameters based on backprop
                        optimizer.step()
                    model.eval()
                    val_loss_lst.append(self.calc_epoch_loss(model,loss,loader_val))
                    train_loss_lst.append(self.calc_epoch_loss(model,loss,loader_train))
                    if train_loss_lst[-1] > self.diverge:
                        print('skip architecture (diverged)')
                        has_diverged = True
                        break
                if not has_diverged:
                    if best_loss > train_loss_lst[-1]:
                        best_loss = train_loss_lst[-1]
                        self.model_state_dict = {k:v for k,v in model.state_dict().items()}
                        self.optimizer_state_dict = {k:v for k,v in optimizer.state_dict().items()}
                        self.optim_str = optim_str

                    self.train_val_plot(train_loss_lst,val_loss_lst,'plots_edge2vec/{}_{}.png'.format(self.method,optim_str))
        best_model = Net(params_dict["architecture"]).type(dtype)
        best_model.set_to_embed(True)
        for param_tensor in self.model_state_dict:
            print(param_tensor, "\t", self.model_state_dict[param_tensor].size())
        # for var_name in self.optimizer_state_dict:
        #     print(var_name, "\t", self.optimizer_state_dict[var_name])
        best_model.load_state_dict(self.model_state_dict)
        if self.save_best_model:
            print('saving model state dictionary...')
            torch.save(best_model.state_dict(), "{}_best_edge_model.pth".format(self.method))
        self.visuallize(best_model,b_test[:,:-1],w_test[:,:-1])

