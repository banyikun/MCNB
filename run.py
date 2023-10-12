from club import CLUB
from locb import LOCB
from cofiba import COFIBA
from sclub import SCLUB
from neuucb_ind import neuucb_ind
from neuucb_one import neuucb_one
from mcnb import meta_ban
import argparse
import numpy as np
import sys 

from load_data import load_movielen_new, load_yelp_new, load_notmnist_mnist_2
from load_data_add import Bandit_multi


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Meta-Ban')
    parser.add_argument('--dataset', default='movie', type=str, help='mnist, yelp, movie')
    parser.add_argument('--method', default='neuucb_ind', type=str, help='locb, club, sclub, cofiba, neuucb_one, neuucb_ind, meta_ban')
    parser.add_argument('--seed', default=0, type=int, help='seed')
    args = parser.parse_args()
    
    data = args.dataset
    additional_datasets = ['fashion','MagicTelescope','mushroom']
    if data == "mnist":
        b = load_notmnist_mnist_2()
        
    elif data == "yelp":
        b = load_yelp_new()
        
    elif data == "movie":
        b = load_movielen_new()
    elif data in additional_datasets:
        b = Bandit_multi(data)
    else:
        print("dataset is not defined. --help")
        sys.exit()
    
    
    method = args.method
    
    if method == "club":   
        model = CLUB(nu = b.num_user, d = b.dim)
        
    elif method == "locb":
        model = LOCB(nu = b.num_user, d = b.dim, gamma = 0.2, num_seeds = 20, delta = 0.1,  detect_cluster = 0)
        
    elif method == "sclub":
        model = SCLUB(nu = b.num_user, d = b.dim)
        
    elif method == "cofiba":
        model = COFIBA(num_users = b.num_user, d = b.dim, num_rounds=10000, L = b.n_arm)
        
    elif method == "neuucb_ind":
        model = neuucb_ind(dim = b.dim, n = b.num_user, n_arm = 10, lr = 0.001)
        
    elif method == "neuucb_one":
        model = neuucb_one(b.dim, lamdba = 0.001, nu = 0.1) 
        
    elif method == "m_cnb":
        if data == "mnist":
            model = meta_ban(dim = b.dim, n = b.num_user, n_arm = 10, gamma = 0.1, lr = 0.0001, user_side = 1)
        elif data in additional_datasets:
            # 0.32
            model = meta_ban(dim = b.dim, n = b.num_user, n_arm = 10, gamma = 0.32, lr = 0.0001, user_side = 1)
        else:
            model = meta_ban(dim = b.dim, n = b.num_user, n_arm = 10, gamma = 0.32, lr = 0.0001)
    else:
        print("method is not defined. --help")
        sys.exit()

       
    print(data, method) 
        
    regrets = []
    summ = 0
    print("Round; Regret; Regret/Round")
    for t in range(10000):
        u, context, rwd = b.step()
        if method == "neuucb_ind" or method == "neuucb_one":
            arm_select, f_res, ucb = model.recommend(u, context, t)
        elif method == "meta_ban":
            arm_select, g, f_res, ucb = model.recommend(u, context, t)
        else:
            arm_select = model.recommend(u, context, t)
        r = rwd[arm_select]
        reg = np.max(rwd) - r
        summ+=reg
        regrets.append(summ)
        if method == "club" or method=="locb":
            model.store_info(i = u, x = context[arm_select], y =r, t = t)
            model.update(i = u, t =t)
        if method=="cofiba":
            model.store_info(i=u, x=context[arm_select], y=r, t=t)
            model.update_cluster(i=u, kk = arm_select, t=t)
        if method == "sclub":
            model.store_info(i = u, x = context[arm_select], y =r, t = t, r = r, br = 1.0 )
            model.split(u, t)
            model.merge(t)
            model.num_clusters[t] = len(model.clusters)
        if method == "neuucb_ind" or method == "neuucb_one":
            model.update(u, context[arm_select], r)
            if t<1000:
                if t%10 == 0:
                    loss = model.train(u, t)
            else:
                if t%100 == 0:
                    loss = model.train(u, t)
        if method == "meta_ban":
            model.update(u, context[arm_select], r, g)
            if t<1000:
                if t%10 == 0:
                    loss = model.train(u, t)
            else:
                if t%100 == 0:
                    loss = model.train(u, t)
            
        if t % 50 == 0:
            print('{}: {:}, {:.4f}'.format(t, summ, summ/(t+1)))
    print("round:", t, "; ", "regret:", summ)
    if args.seed:
        np.save("regret/{}_{}_regret_{}".format(args.dataset, args.method, args.seed),  regrets)
    else:
        np.save("regret/{}_{}_regret".format(args.dataset, args.method),  regrets)
    
    
    
    
    
    
