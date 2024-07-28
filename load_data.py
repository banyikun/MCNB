from sklearn.datasets import fetch_openml
from sklearn.utils import shuffle
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import normalize
import numpy as np
import pandas as pd 
from scipy.stats import norm
import torch
import torchvision
from torchvision import datasets, transforms
from sklearn.cluster import KMeans
from collections import defaultdict

    
    
class load_movielen_dif_user():
    def __init__(self, n_users):
        # Fetch data
        self.m = np.load("./movie_2000users_10000items_entry.npy")
        self.U = np.load("./movie_2000users_10000items_features.npy")
        self.I = np.load("./movie_10000items_2000users_features.npy")
        kmeans = KMeans(n_clusters=n_users, random_state=0).fit(self.U)
        self.groups = kmeans.labels_
        self.n_arm = 10
        self.dim = 20
        self.num_user = n_users
        self.pos_index = defaultdict(list)
        self.neg_index = defaultdict(list)
        for i in self.m:
            if i[2] ==1:
                self.pos_index[self.groups[i[0]]].append((i[0], i[1]))
            else:
                self.neg_index[self.groups[i[0]]].append((i[0], i[1]))   


    def step(self):    
        u = np.random.choice(range(self.num_user))
        g = self.groups[u]
        arm = np.random.choice(range(10))
        #print(pos_index.shape)
        p_d = len(self.pos_index[g])
        n_d = len(self.neg_index[g])
        pos = np.array(self.pos_index[g])[np.random.choice(range(p_d), 9, replace=True)]
        neg = np.array(self.neg_index[g])[np.random.choice(range(n_d), 1, replace=True)]
        X_ind = np.concatenate((pos[:arm], neg, pos[arm:]), axis=0)
        X = []
        for ind in X_ind:
            X.append( np.concatenate( (self.I[ind[1]], self.U[ind[0]]), axis=None)    )
        rwd = np.zeros(self.n_arm)
        rwd[arm] = 1
        contexts = norm.pdf(np.array(X), loc=0, scale=0.5)
        return g, contexts, rwd 