import networkx as nx
import numpy as np
from utlis import edge_probability, is_power2
import sys





def isInvertible(S):
    return np.linalg.cond(S) < 1 / sys.float_info.epsilon


class Base:
    # Base agent for online clustering of bandits
    def __init__(self, d):
        self.d = d

    def _beta(self, N, t):
        return np.sqrt(self.d * np.log(1 + N / self.d) + 4 * np.log(t) + np.log(2)) + 1

    def _select_item_ucb(self, S, Sinv, theta, items, N, t):
        return np.argmax(np.dot(items, theta) + self._beta(N, t) * (np.matmul(items, Sinv) * items).sum(axis = 1))

    def _update_inverse(self, S, b, Sinv, x, t):
        Sinv = np.linalg.inv(S)
        theta = np.matmul(Sinv, b)
        return Sinv, theta


class LinUCB_IND(Base):
    # each user is an independent LinUCB
    def __init__(self, nu, d):
        super(LinUCB_IND, self).__init__(d)
        self.S = {i:np.eye(d) for i in range(nu)}
        self.b = {i:np.zeros(d) for i in range(nu)}
        self.Sinv = {i:np.eye(d) for i in range(nu)}
        self.theta = {i:np.zeros(d) for i in range(nu)}

        self.N = np.zeros(nu)

    def recommend(self, i, items, t):
        return self._select_item_ucb(self.S[i], self.Sinv[i], self.theta[i], items, self.N[i], t)

    def store_info(self, i, x, y, t):
        self.S[i] += np.outer(x, x)
        self.b[i] += y * x
        self.N[i] += 1

        self.Sinv[i], self.theta[i] = self._update_inverse(self.S[i], self.b[i], self.Sinv[i], x, self.N[i])



class Cluster:
    def __init__(self, users, S, b, N):
        self.users = users # a list/array of users
        self.S = S
        self.b = b
        self.N = N
        self.Sinv = np.linalg.inv(self.S)
        self.theta = np.matmul(self.Sinv, self.b)

class CLUB(LinUCB_IND):
    # random_init: use random initialization or not
    def __init__(self, nu, d, T = 10000, edge_probability = 1):
        super(CLUB, self).__init__(nu, d)
        self.nu = nu
        # self.alpha = 4 * np.sqrt(d) # parameter for cut edge
        self.G = nx.complete_graph(nu)
        self.clusters = {0:Cluster(users=range(nu), S=np.eye(d), b=np.zeros(d), N=0)}
        self.cluster_inds = np.zeros(nu)

        self.num_clusters = np.zeros(T)

    def recommend(self, i, items, t):
        cluster = self.clusters[self.cluster_inds[i]]
        return self._select_item_ucb(cluster.S, cluster.Sinv, cluster.theta, items, cluster.N, t)

    def store_info(self, i, x, y, t):
        super(CLUB, self).store_info(i, x, y, t)

        c = self.cluster_inds[i]
        self.clusters[c].S += np.outer(x, x)
        self.clusters[c].b += y * x
        self.clusters[c].N += 1

        self.clusters[c].Sinv, self.clusters[c].theta = self._update_inverse(self.clusters[c].S, self.clusters[c].b, self.clusters[c].Sinv, x, self.clusters[c].N)

    def _if_split(self, theta, N1, N2):
        # alpha = 2 * np.sqrt(2 * self.d)
        alpha = 1
        def _factT(T):
            return np.sqrt((1 + np.log(1 + T)) / (1 + T))
        return np.linalg.norm(theta) >  alpha * (_factT(N1) + _factT(N2))
 
    def update(self, i, t):
        update_clusters = False
      
        c = self.cluster_inds[i]

        A = [a for a in self.G.neighbors(i)]
        for j in A:
            if self.N[i] and self.N[j] and self._if_split(self.theta[i] - self.theta[j], self.N[i], self.N[j]):
                self.G.remove_edge(i, j)
                #print(i,j)
                update_clusters = True

        if update_clusters:
            C = set()
            C = nx.node_connected_component(self.G, i)
            if len(C) < len(self.clusters[c].users):
                remain_users = set(self.clusters[c].users)
                self.clusters[c] = Cluster(list(C), S=sum([self.S[k]-np.eye(self.d) for k in C])+np.eye(self.d), b=sum([self.b[k] for k in C]), N=sum([self.N[k] for k in C]))

                remain_users = remain_users - set(C)
                c = max(self.clusters) + 1
                while len(remain_users) > 0:
                    j = np.random.choice(list(remain_users))
                    C = nx.node_connected_component(self.G, j)

                    self.clusters[c] = Cluster(list(C), S=sum([self.S[k]-np.eye(self.d) for k in C])+np.eye(self.d), b=sum([self.b[k] for k in C]), N=sum([self.N[k] for k in C]))
                    for j in C:
                        self.cluster_inds[j] = c

                    c += 1
                    remain_users = remain_users - set(C)
            
        self.num_clusters[t] = len(self.clusters)
        