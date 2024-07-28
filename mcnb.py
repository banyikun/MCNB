from packages import *
#from load_data import load_yelp, load_mnist_1d, load_movielen

if torch.cuda.is_available():  
    dev = "cuda:0" 
else:  
    dev = "cpu" 
device = torch.device(dev)


class Network_u(nn.Module):
    def __init__(self, dim, hidden_size=100):
        super(Network_u, self).__init__()
        self.fc1 = nn.Linear(dim, hidden_size)
        self.activate = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
    def forward(self, x):
        return self.fc2(self.activate(self.fc1(x)))

    
    
class MCNB:
    def __init__(self, dim, n, n_arm, gamma, lr = 0.01, hidden=100, nu = 0.001):
        self.lr = lr
        self.dim = dim
        self.hidden = hidden
        self.t = 0
        self.gamma = gamma # for clustering
        self.nu = nu # for exploration
        self.g = [] # current group
        self.n = n #number of users
        self.u_count = [0]*n # the number of serving for a user
        self.users = range(n)
        self.u_fun = {} # neural networks of users
        self.meta_fun = Network_u(self.dim, hidden_size=self.hidden).to(device) # meta neural network
        for i in range(n):
            self.u_fun[i] = Network_u(dim, hidden_size=hidden).to(device) # user neural network        
        self.contexts = defaultdict(list)
        self.rewards = defaultdict(list)
        

    def update(self, u, context, reward, g):
        self.contexts[u].append(torch.from_numpy(context.reshape(1, -1)).float())
        self.rewards[u].append(reward)

            
    def train_meta(self, g, t, train_limit, meta_lr, y=1):
        if y == 0:
            optimizer = optim.SGD(self.meta_fun.parameters(), lr=meta_lr)
        else:
            optimizer = optim.Adam(self.meta_fun.parameters(), lr=meta_lr)
        index = []
        for u in g:
            for j in range(len(self.rewards[u])):
                index.append((u, j))
        length = len(index)
        cnt = 0
        if length >0:
            tot_loss = 0
            while True:
                batch_loss = 0
                np.random.shuffle(index)
                for idx in index:
                    u = idx[0]
                    arm = idx[1]
                    c = self.contexts[u][arm]
                    r = self.rewards[u][arm]
                    optimizer.zero_grad()
                    loss = (self.meta_fun(c.to(device)) - r)**2                

                    loss.backward()
                    optimizer.step()

                    batch_loss += loss.item()
                    tot_loss += loss.item()
                    cnt += 1
                    if cnt >= train_limit:
                        return tot_loss / cnt
                if batch_loss / length <= 1e-3:
                    return batch_loss / length

                
                
    def get_group(self, u, context):
        u_pred = self.u_fun[u](context)
        g = set([u])
        for i in self.users:
            diff = abs(self.u_fun[i](context) -  u_pred)
            if diff < self.gamma:
                g.add(i)
        time_b = time.time()
        g_limit = int(self.n/5)
        if len(g) > g_limit:
            g = set(np.random.choice(list(g), g_limit))
            g.add(u)
            return g
        else:
            return g
       

    def select(self, u, context, t):
        self.t = t
        ucb_list = []
        self.u_count[u] += 1
        for c in context:
            c_tensor = torch.from_numpy(c).float().to(device)
            if t%10 == 0: # meta adapation, %10 to accelerate 
                self.g = self.get_group(u,  c_tensor)
                self.train_meta(self.g, t, 100, 1e-2, 0)
            res = self.meta_fun(c_tensor)
            self.meta_fun.zero_grad()
            res.backward()
            gra = torch.cat([p.grad.flatten().detach() for p in self.meta_fun.parameters()]) # gradient
            sigma =torch.sum(self.nu * gra * gra).item() 
            ucb = res.item() + np.sqrt(sigma) + np.sqrt(1/self.u_count[u]) * self.nu 
            ucb_list.append(ucb)
        arm = np.argmax(ucb_list)
        return arm, self.g, ucb_list
    
    
    
    def train(self, u, t):
        optimizer = optim.Adam(self.u_fun[u].parameters(), lr=self.lr)
        length = len(self.rewards[u])
        index = np.arange(length)
        np.random.shuffle(index)
        cnt = 0
        tot_loss = 0
        if length >0:
            while True:
                batch_loss = 0
                for idx in index:
                    c = self.contexts[u][idx]
                    r = self.rewards[u][idx]
                    optimizer.zero_grad()
                    loss = (self.u_fun[u](c.to(device)) - r)**2                
                    loss.backward()
                    optimizer.step()
                    batch_loss += loss.item()
                    tot_loss += loss.item()
                    cnt += 1
                    if cnt >= 500:
                        return tot_loss / cnt
                if batch_loss / length <= 1e-3:
                    return batch_loss / length
        else:
            return 0.0
            

