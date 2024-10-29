# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 15:39:58 2021

@author: ZHU Haoren
"""

import numpy as np
import torch
import random
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from data_provider import TripletUniformPair

from utils import delete_item_in_history




def find_negative(history, item_space):
    while True:
        neg_item = torch.randint(0, item_space, (1,))[0]
        if neg_item not in history:
            return neg_item


class POP():
    """
    Recommend the most popular items
    """
    def __init__(self, item_space):
        self.item_space = item_space
        self.item = torch.zeros(item_space)
        
    def train(self, train, valid=None):
        # Obtain the sequence datta
        for i in train:
            for j in i:
                item_id = j - 1
                self.item[item_id] += 1
        _, self.sorted_idx = self.item.sort(descending= True)
        self.sorted_idx += 1
    
    def save(self, path="./baselines/pop/"):
        if not os.path.isdir(path):
            os.makedirs(path)
        print("Saving model...")
        torch.save(self.sorted_idx, path + 'sorted_idx.pt')
        
    def load(self, path="./baselines/pop/"):
        print("Loading model...")
        self.sorted_idx = torch.load(path + 'sorted_idx.pt')
                
    def predict_next(self, seqs, users=None, top_k = 20, use_h=True):
        """
        Predict the item set at the next place
        """
        preds =  torch.zeros((len(seqs), top_k))
            
        for i in range(len(seqs)):
            seq = seqs[i]
            indices = self.sorted_idx.clone()
            if use_h == True:
                indices = delete_item_in_history(indices, torch.tensor(seq))

            preds[i][:] = indices[:top_k]
                
        return preds
    
class MC():
    def __init__(self, config):
        self.item_space = config.n_item
        self.item_successor = [set() for _ in range(self.item_space)]
        self.lam = config.lam
        self.K = config.K
        self.learn_rate = config.lr
        self.max_iter = config.max_iter
        self.gam = torch.rand(self.item_space, self.K, dtype=torch.float) - 0.5
        self.eta = torch.rand(self.K, self.item_space, dtype=torch.float) - 0.5
        self.earlystop_threshold = config.earlystop_threshold
        self.mc = None
        self.path = "./baselines/" + config.dataset + "/" + config.method + "/"

        
    def find_item(self):
        while True:
            item = torch.randint(0, self.item_space, (1,))[0]
            if len(self.item_successor[item]) > 0:
                return item
    
    def find_neg_suc(self, item):
        while True:
            neg_item = torch.randint(0, self.item_space, (1,))[0]
            if neg_item.item() not in self.item_successor[item]:
                return neg_item
            
    def save(self, path="./baselines/mc/"):
        if not os.path.isdir(path):
            os.makedirs(path)
        print("Saving model...")
        torch.save(self.gam, path + 'gam.pt')
        torch.save(self.eta, path + 'eta.pt')
        
    def load(self, path="./baselines/mc/"):
        print("Loading model...")
        self.gam = torch.load(path + 'gam.pt')
        self.eta = torch.load(path + 'eta.pt')
        
    def train(self, train, valid=None):
        print("Start Training...\n")
        train = [np.array(seq) - 1 for seq in train]
        for seq in train:
            for i in range(len(seq)-1):
                pre = seq[i]
                suc = seq[i+1]
                self.item_successor[pre].add(suc)

        # Model parameters
        num_relation = sum([len(_) for _ in self.item_successor])
        gam = self.gam
        eta = self.eta
        lam = self.lam
        learn_rate = self.learn_rate
        # Early stop configuration
        previous = None
        
        for it in range(self.max_iter):
            objective = 0
            regularization = 0
            
            for ind in range(num_relation):
                u = self.find_item()
                i = random.choice(list(self.item_successor[u]))
                j = self.find_neg_suc(u)
                z = torch.sigmoid(torch.dot(gam[u,:],eta[:,i]) - torch.dot(gam[u,:],eta[:,j]))

                gam[u,:] += learn_rate*((1-z)*(eta[:,i]-eta[:,j])-2*lam*gam[u,:])
                eta[:,i] += learn_rate*((1-z)*(gam[u,:]) - 2*lam*eta[:,i])
                eta[:,j] += learn_rate*((1-z)*(-gam[u,:]) - 2*lam*eta[:,j])    
                objective += torch.log(z)
            
            regularization = objective - lam*torch.sum(torch.square(gam)) - lam*torch.sum(torch.square(eta))        
            if it%2 == 0:
                print("Iteration %i: %f"%(it, regularization))
                if it >= 10:
                    self.gam = gam
                    self.eta = eta
                    self.save(self.path)
                
            # Early stop handler
            if previous == None:
                previous = regularization
            elif abs(previous - regularization) < self.earlystop_threshold:
                print("Objective: %f. Early stopping..."%regularization)
                break
            else:
                previous = regularization
            
            
        print("End Training...\n")
              
    def predict_next(self, seqs, users = None,  top_k = 50, use_h = True):
        if self.mc == None:
            self.mc = torch.matmul(self.gam, self.eta)

        preds =  torch.zeros((len(seqs), top_k))
        for i in range(len(seqs)):
            seq = seqs[i]
            last = int(seq[-1] - 1)
            _, indices = self.mc[last].sort(descending = True)# Obtain the top-k item list
            indices += 1
            if use_h == True:
                indices = delete_item_in_history(indices, torch.tensor(seq))

            preds[i][:] = indices[:top_k]
        return preds 
    
class FPMC():
    def __init__(self, config):
        self.user_count = config.n_user
        self.item_space = config.n_item
        self.item_successor = [[] for _ in range(self.item_space)]
        self.lam = config.lam
        self.K1 = config.K1
        self.K2 = config.K2
        self.learn_rate = config.lr
        self.max_iter = config.max_iter
        self.gamU = torch.rand(self.user_count, self.K1) - 0.5
        self.gamI = torch.rand(self.K1, self.item_space) - 0.5
        self.kap = torch.rand(self.item_space, self.K2) - 0.5
        self.eta = torch.rand(self.K2, self.item_space) - 0.5
        self.earlystop_threshold = config.earlystop_threshold
        self.a = None
        self.b = None
        self.path = "./baselines/" + config.dataset + "/" + config.method + "/"
            
    def normalization(self, it):
        dist = torch.sqrt(torch.sum(torch.square(self.H[it,:])))
        if dist > 1:
            self.H[it,:] = self.H[it,:] / dist
            
    def save(self, path):
        if not os.path.isdir(path):
            os.makedirs(path)
        print("Saving model...")
        torch.save(self.gamU, path + 'gamu.pt')
        torch.save(self.gamI, path + 'gami.pt')
        torch.save(self.kap, path + 'kap.pt')
        torch.save(self.eta, path + 'eta.pt')
        
    def load(self, path):
        print("Loading model...")
        self.gamU = torch.load(path + 'gamu.pt')
        self.gamI = torch.load(path + 'gami.pt')
        self.kap = torch.load(path + 'kap.pt')
        self.eta = torch.load(path + 'eta.pt')
        
    def train(self, train, valid=None):
        # Obtain the sequence datta
        train = [np.array(seq) - 1 for seq in train]
        num_relation = 0
        for seq in train:
            for i in range(len(seq)-1):
                num_relation += 1
        # Early stop configuration
        previous = None
        
        for it in range(self.max_iter):
            objective = 0
            regularization = 0
            for ind in range(num_relation):
                # Random select a user with sufficiently long sequences
                while True:
                    u = torch.randint(0, self.user_count, (1,))[0]
                    if len(train[u]) > 1:
                        break
                # Select the user sequence
                seq = train[u]
                position = torch.randint(0, len(seq)-1, (1,))[0]
                p = seq[position]        # previous item
                i = seq[position + 1]    # positive item
                j = find_negative(train[u], self.item_space)  # negative item
                
                z = torch.sigmoid(torch.dot(self.gamU[u,:],self.gamI[:,i]) - \
                                  torch.dot(self.gamU[u,:],self.gamI[:,j]) + \
                                  torch.dot(self.kap[p,:],self.eta[:,i]) - \
                                  torch.dot(self.kap[p,:],self.eta[:,j]))

                self.gamU[u,:] += self.learn_rate*((1-z)*(self.gamI[:,i]-self.gamI[:,j]) - 2*self.lam*self.gamU[u,:])
                self.gamI[:,i] += self.learn_rate*((1-z)*(self.gamU[u,:]) - 2*self.lam*self.gamI[:,i])
                self.gamI[:,j] += self.learn_rate*((1-z)*(-self.gamU[u,:]) - 2*self.lam*self.gamI[:,j])
                self.kap[p,:] =+ self.learn_rate*((1-z)*(self.eta[:,i]-self.eta[:,j])-2*self.lam*self.kap[p,:])
                self.eta[:,i] += self.learn_rate*((1-z)*(self.kap[p,:]) - 2*self.lam*self.eta[:,i])
                self.eta[:,j] += self.learn_rate*((1-z)*(-self.kap[p,:]) - 2*self.lam*self.eta[:,j])
                
                objective += torch.log(z)
        
                regularization = objective - self.lam*torch.sum(torch.square(self.gamU)) - \
                                             self.lam*torch.sum(torch.square(self.gamI)) - \
                                             self.lam*torch.sum(torch.square(self.kap)) - \
                                             self.lam*torch.sum(torch.square(self.eta))
            if it%2 == 0:
                print("Iteration %i: %f"%(it, regularization))
                
            # Early stop handler
            if previous == None:
                previous = regularization
            elif abs(previous - regularization) < self.earlystop_threshold:
                print("Objective: %f. Early stopping..."%regularization)
                break
            else:
                previous = regularization
                
            if it >= 10:
                self.save(self.path)
    
    def predict_next(self, seqs, users, top_k = 50, use_h = True):
        if self.a == None:
            self.a = torch.matmul(self.gamU, self.gamI)
            self.b = torch.matmul(self.kap, self.eta)
            

        preds =  torch.zeros((len(seqs), top_k))
            
        for i in range(len(seqs)):
            seq = seqs[i]
            last = int(seq[-1] - 1)
            l = self.a[users[i],:]+ self.b[last,:]
            _, indices = l.sort(descending = True)# Obtain the top-k item list
            indices += 1
            if use_h == True:
                indices = delete_item_in_history(indices, torch.tensor(seq))

            preds[i][:] = indices[:top_k]
        return preds         

class TransRec():
    def __init__(self, config):
        self.user_count = config.n_user
        self.item_space = config.n_item
        self.item_successor = [[] for _ in range(self.item_space)]
        self.lam = config.lam
        self.bias_lam = config.bias_lam
        self.reg_lam = config.reg_lam
        self.K = config.K
        self.learn_rate = config.lr
        self.max_iter = config.max_iter
        self.r = torch.zeros(self.K)
        self.R = torch.zeros((self.user_count, self.K))
        self.H = torch.rand(self.item_space, self.K) - 0.5
        self.beta = torch.zeros(self.item_space)
        self.earlystop_threshold = config.earlystop_threshold
        self.path = "./baselines/" + config.dataset + "/" + config.method + "/"
    
    def find_neg(self, item):
        while True:
            neg_item = torch.randint(0, self.item_space, (1,))[0]
            if neg_item != item:
                return neg_item
            
    def normalization(self, it):
        dist = torch.sqrt(torch.sum(torch.square(self.H[it,:])))
        if dist > 1:
            self.H[it,:] = self.H[it,:] / dist
            
    def save(self, path):
        if not os.path.isdir(path):
            os.makedirs(path)
        print("Saving model...")
        torch.save(self.r, path + 'r.pt')
        torch.save(self.R, path + 'rr.pt')
        torch.save(self.H, path + 'h.pt')
        torch.save(self.beta, path + 'beta.pt')
        
    def load(self, path):
        print("Loading model...")
        self.r = torch.load(path + 'r.pt')
        self.R = torch.load(path + 'rr.pt')
        self.H = torch.load(path + 'h.pt')
        self.beta = torch.load(path + 'beta.pt')
        
    def train(self, train, valid=None):
        train = [np.array(seq) - 1 for seq in train]
        num_relation = 0
        for seq in train:
            for i in range(len(seq)-1):
                num_relation += 1
        # Early stop configuration
        previous = None
        
        for it in range(self.max_iter):
            objective = 0
            regularization = 0
            
            for ind in range(num_relation):
                # Random select a user with sufficiently long sequences
                while True:
                    u = torch.randint(0, self.user_count, (1,))[0]
                    if len(train[u]) > 1:
                        break
                # Select the user sequence
                seq = train[u]
                position = torch.randint(0, len(seq)-1, (1,))[0]
                p = seq[position]        # previous item
                i = seq[position + 1]    # positive item
                j = find_negative(train[u], self.item_space)                 # negative item
                
                d1 = self.H[p,:] + self.r + self.R[u,:] - self.H[i,:]
                d2 = self.H[p,:] + self.r + self.R[u,:] - self.H[j,:]
                
                z = torch.sigmoid(-self.beta[i] + self.beta[j] - \
                            torch.sum(torch.square(d1)) + \
                            torch.sum(torch.square(d2)))

                self.beta[i] += self.learn_rate*(-(1-z) - 2*self.bias_lam*self.beta[i])
                self.beta[j] += self.learn_rate*((1-z) - 2*self.bias_lam*self.beta[j])
                self.H[p,:] += self.learn_rate*((1-z)*2*(d2-d1) - 2*self.lam*self.H[p,:])
                self.H[i,:] += self.learn_rate*((1-z)*2*(d1) - 2*self.lam*self.H[i,:])
                self.H[j,:] += self.learn_rate*((1-z)*2*(-d2) - 2*self.lam*self.H[j,:])
                self.r += self.learn_rate*((1-z)*2*(d2-d1) - 2*self.lam*self.r)
                self.R[u] = self.learn_rate*((1-z)*2*(d2-d1) - 2*self.reg_lam*self.R[u])
                
                self.normalization(p)
                self.normalization(i)
                self.normalization(j)
                
                objective += torch.log(z)
            
            regularization = objective - self.lam*torch.sum(torch.square(self.H)) - \
                                         self.lam*torch.sum(torch.square(self.r)) - \
                                         self.reg_lam*torch.sum(torch.square(self.R)) - \
                                         self.bias_lam*torch.sum(torch.square(self.beta))
            if it%2 == 0:
                print("Iteration %i: %f"%(it, regularization))
                
            # Early stop handler
            if previous == None:
                previous = regularization
            elif abs(previous - regularization) < self.earlystop_threshold:
                print("Objective: %f. Early stopping..."%regularization)
                break
            else:
                previous = regularization
                
            if it >= 10:
                self.save(self.path)

    def predict_next(self, seqs, users, top_k = 50, use_h = True):

        preds =  torch.zeros((len(seqs), top_k))
        for i in range(len(seqs)):
            seq = seqs[i]
            last = int(seq[-1] - 1)
            a = self.H[last]
            c = self.R[users[i], :]
            l = torch.square(a.repeat(self.item_space, 1) + self.r.repeat(self.item_space, 1) + \
                             c.repeat(self.item_space, 1) - self.H)
            l = -torch.sum(l, 1) - self.beta
            _, indices = l.sort(descending = True)# Obtain the top-k item list
            indices += 1
            if use_h == True:
                indices = delete_item_in_history(indices, torch.tensor(seq))

            preds[i][:] = indices[:top_k]
        return preds        
    
class BPR(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_user = config.n_user
        self.n_item = config.n_item
        self.W = nn.Parameter(torch.empty(self.n_user, config.dim))
        self.H = nn.Parameter(torch.empty(self.n_item, config.dim))
        nn.init.xavier_normal_(self.W.data)
        nn.init.xavier_normal_(self.H.data)
        self.weight_decay = config.weight_decay
        self.lr = config.lr
        self.n_epochs=config.n_epochs
        self.batch_size = config.batch_size
        self.earlystop_threshold = config.earlystop_threshold
        self.path = "./baselines/" + config.dataset + "/" + config.method + "/"

    def save(self, path):
        if not os.path.isdir(path):
            os.makedirs(path)
        print("Saving model...")
        torch.save(self.state_dict(), path + 'bpr.pt')
        
    def load(self, path):
        print("Loading model...")
        model_info = torch.load(path + 'bpr.pt')
        self.load_state_dict(model_info)

    def forward(self, u, i, j):
        """Return loss value.
        
        Args:
            u(torch.LongTensor): tensor stored user indexes. [batch_size,]
            i(torch.LongTensor): tensor stored item indexes which is prefered by user. [batch_size,]
            j(torch.LongTensor): tensor stored item indexes which is not prefered by user. [batch_size,]
        
        Returns:
            torch.FloatTensor
        """
        u = self.W[u, :]
        i = self.H[i, :]
        j = self.H[j, :]
        x_ui = torch.mul(u, i).sum(dim=1)
        x_uj = torch.mul(u, j).sum(dim=1)
        x_uij = x_ui - x_uj
        log_prob = F.logsigmoid(x_uij).sum()
        regularization = self.weight_decay * (u.norm(dim=1).pow(2).sum() + i.norm(dim=1).pow(2).sum() + j.norm(dim=1).pow(2).sum())
        return -log_prob + regularization
    
    def find_neg(self, item):
        while True:
            neg_item = torch.randint(0, self.item_space, (1,))[0]
            if neg_item != item:
                return neg_item
    
    def train(self, train, valid=None):
        # Obtain the triplet pair
        train = [np.array(seq) - 1 for seq in train]
        counter = 0
        pair = []
        for u_id in range(len(train)):
            seq = train[u_id]
            for j in seq:
                pair.append(torch.LongTensor([u_id, j]))
                counter +=1
        
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        dataset = TripletUniformPair(self.n_item, train, pair, True, self.n_epochs)
        loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=0)
        
        # Early stop configuration
        previous = None
        smooth_loss = 0
        idx = 0
        for u, i, j in loader:
            optimizer.zero_grad()
            loss = self(u, i, j)
            loss.backward()
            optimizer.step()
            smooth_loss = smooth_loss*0.99 + loss*0.01
            if idx % 500 == 0:
                print('loss: %.4f, ' % smooth_loss)
                
            idx += 1
            
            # Early stop handler
            n_batch = len(pair)//self.batch_size
            if idx >=  n_batch* 10 and idx%n_batch==0:
                self.save(self.path)
                # Early stop handler
                if previous == None:
                    previous = smooth_loss
                elif abs(previous - smooth_loss) < self.earlystop_threshold:
                    print("Objective: %f. Early stopping..."%smooth_loss)
                    break
                else:
                    previous = smooth_loss

    def predict_next(self, seqs, users, top_k = 50, use_h = True):
        """Return recommended item list given users.
        Args:
            u(torch.LongTensor): tensor stored user indexes. [batch_size,]
        Returns:
            pred(torch.LongTensor): recommended item list sorted by preference. [batch_size, item_size]
        """

        preds =  torch.zeros((len(seqs), top_k))
            
        users = torch.LongTensor(users)
        u = self.W[users, :]
        x_ui = torch.mm(u, self.H.t())
        _, items = x_ui.sort(descending = True, dim=1)# Obtain the top-k item list
        items += 1
            
        for i in range(len(users)):  
            indices = items[i]
            if use_h == True:
                seq = seqs[i]
                indices = delete_item_in_history(indices, torch.tensor(seq))

            preds[i][:] = indices[:top_k]
        return preds        
            
                