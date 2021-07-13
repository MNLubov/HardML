#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math

import numpy as np
import torch
from catboost.datasets import msrank_10k
from sklearn.preprocessing import StandardScaler

from typing import List


# In[2]:


class ListNet(torch.nn.Module):
    def __init__(self, num_input_features, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.model = torch.nn.Sequential(
        torch.nn.Linear(num_input_features, self.hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(self.hidden_dim, 1)
        )

    def forward(self, input_1: torch.Tensor) -> torch.Tensor:
        logits = self.model(input_1)
        return logits


# In[3]:


class Solution:
    def __init__(self, n_epochs = 5, listnet_hidden_dim = 30,
                 lr = 0.001, ndcg_top_k = 10):
        self._prepare_data()
        self.num_input_features = self.X_train.shape[1]
        self.ndcg_top_k = ndcg_top_k
        self.n_epochs = n_epochs

        self.model = self._create_model(
             self.num_input_features, listnet_hidden_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)


# In[4]:


class Solution(Solution):
    def _get_data(self): #-> List[np.ndarray]:
        train_df, test_df = msrank_10k()

        X_train = train_df.drop([0, 1], axis=1).values
        y_train = train_df[0].values
        query_ids_train = train_df[1].values.astype(int)

        X_test = test_df.drop([0, 1], axis=1).values
        y_test = test_df[0].values
        query_ids_test = test_df[1].values.astype(int)

        return [X_train, y_train, query_ids_train, X_test, y_test, query_ids_test]
    
    def _prepare_data(self):
        (X_train, y_train, self.query_ids_train,
            X_test, y_test, self.query_ids_test) = self._get_data()
        X_train = self._scale_features_in_query_groups(X_train, self.query_ids_train)
        X_test = self._scale_features_in_query_groups(X_test, self.query_ids_test)
        self.X_train = torch.FloatTensor(X_train) #torch.from_numpy(X_train)
        self.X_test = torch.FloatTensor(X_test)
        self.ys_train = torch.FloatTensor(y_train)
        self.ys_test = torch.FloatTensor(y_test)
       
    def _scale_features_in_query_groups(self, inp_feat_array, inp_query_ids):
        scaler = StandardScaler()
        ids_unique = np.unique(inp_query_ids)
    
        for ids in ids_unique:
            ids_index = np.argwhere(inp_query_ids == ids).flatten()
            x = inp_feat_array[ids_index]
            x = scaler.fit_transform(x)
            inp_feat_array[ids_index] = x
            
        return inp_feat_array   


# In[5]:


class Solution(Solution):
    def _create_model(self, listnet_num_input_features, 
                      listnet_hidden_dim): 
        torch.manual_seed(0)
        net = ListNet(listnet_num_input_features, listnet_hidden_dim)
        return net


# In[6]:


class Solution(Solution):
    def fit(self): # -> List[float]:
        ndcg_list = []
        
        for epoch in range(self.n_epochs):
            self._train_one_epoch()
            ndcg_epoch = self._eval_test_set()
            print(epoch, ndcg_epoch)
            ndcg_list.append(ndcg_epoch)
        
        return ndcg_list


# In[7]:


class Solution(Solution):
    def _calc_loss(self, batch_ys, batch_pred):
        pbatch_ys = torch.softmax(batch_ys, dim = 0)
        pbatch_pred = torch.softmax(batch_pred, dim = 0)
        return -torch.sum(pbatch_ys * torch.log(pbatch_pred)) 
        #return -torch.sum(pbatch_ys * torch.log(pbatch_pred / pbatch_ys)) 

    def _train_one_epoch(self):# -> None:
        self.model.train()
        
        ids_unique = np.random.permutation(np.unique(self.query_ids_train))

        i = 0
        for ids in ids_unique:
            self.optimizer.zero_grad()
            ids_index = np.argwhere(self.query_ids_train == ids).flatten()
            batch_X = self.X_train[ids_index]
            batch_ys = torch.reshape(self.ys_train[ids_index], (self.ys_train[ids_index].shape[0], 1))
            
            batch_pred = self.model(batch_X)
            batch_loss = self._calc_loss(batch_ys, batch_pred)
            batch_loss.backward(retain_graph = True)
            if i % 10 == 0:
                print(batch_loss)
            i += 1
            self.optimizer.step()


# In[8]:


class Solution(Solution):
    def _eval_test_set(self): # -> float:
        with torch.no_grad():
            self.model.eval()
            ndcgs = []
            
            ids_unique = np.unique(self.query_ids_test)
    
            for ids in ids_unique:
                ids_index = np.argwhere(self.query_ids_test == ids).flatten()
                test_pred = self.model(self.X_test[ids_index])
                test_true = torch.reshape(self.ys_test[ids_index], (self.ys_test[ids_index].shape[0], 1))
                ndcg = self._ndcg_k(test_true, test_pred, self.ndcg_top_k)
                ndcgs.append(ndcg)

            return np.mean(ndcgs)

    def _ndcg_k(self, ys_true, ys_pred, ndcg_top_k): #-> float:
        #sum of gains
        DCG_k = 0.0
        IDCG_k = 0.0
           
        #sorting ys_pred in descending order and obtaining indices for ys_true 
        ys_pred_sort, indices = torch.sort(ys_pred, descending = True)
        ys_true_sort, indices2 = torch.sort(ys_true, descending = True)
        
        if len(ys_true) < ndcg_top_k:
            ndcg_top_k = len(ys_true)  
            
        #calculation of discounted cumulative gain
        for k in range(0, ndcg_top_k):
            DCG_k = DCG_k + (2**(ys_true[indices[k]]) - 1) / math.log2(k + 2)
            IDCG_k = IDCG_k + (2**(ys_true[indices2[k]]) - 1) / math.log2(k + 2)

        if IDCG_k == 0.0:
            NDCG_k = 0.0
            #print(IDCG)
        else:
            NDCG_k = DCG_k / IDCG_k
            #print(NDCG_k)

        #if np.isnan(NDCG_k):
        #    NDCG_k = 0.0
    
        return NDCG_k


# In[ ]:




