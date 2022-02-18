from packages import *

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

    
    
class meta_ban:
    def __init__(self, dim, n, n_arm, gamma, lr = 0.01, hidden=100, lamdba = 0.001, nu = 0.01, user_side = 0):
        self.context_list = []
        self.reward = []
        self.lr = lr
        self.dim = dim
        self.hidden = hidden
        self.t = 0
        self.meta_lr = lr
        
        self.gamma = gamma
        self.lamdba = lamdba
        self.nu = nu
        self.g = []
        self.s_g = []
        self.user_side = user_side
        
        self.users = range(n)
        self.u_funcs = {}
        self.u_his = {}
        
        self.gmodel = Network_u(self.dim, hidden_size=self.hidden).to(device)
        
        
        for i in range(n):
            self.u_funcs[i] = Network_u(dim, hidden_size=hidden).to(device)

        self.cur_group = []
        
        self.contexts = defaultdict(list)
        self.rewards = defaultdict(list)
        
            
    def get_group(self, u, tensor):
        g = []
        u_pred = self.u_funcs[u](tensor)
        for i in self.users:
            if abs(self.u_funcs[i](tensor) -  u_pred) < self.gamma:
                g.append(i)
        if len(g) > 4:
            return np.random.choice(list(g), 4)
        else: 
            return g
    
    
    def update(self, u, context, reward, g):
        for i in g:
            self.contexts[i].append(torch.from_numpy(context.reshape(1, -1)).float())
            self.rewards[i].append(reward)

            
    def train_meta(self, g, t):
        optimizer = optim.Adam(self.gmodel.parameters(), lr=self.lr)
        index = []
        for u in g:
            for j in range(len(self.rewards[u])):
                index.append((u, j))
          
        length = len(index)
        if length >0:
            np.random.shuffle(index)
            cnt = 0
            tot_loss = 0
            while True:
                batch_loss = 0
                for idx in index:
                    u = idx[0]
                    arm = idx[1]
                    c = self.contexts[u][arm]
                    r = self.rewards[u][arm]
                    optimizer.zero_grad()
                    loss = (self.gmodel(c.to(device)) - r)**2                

                    loss.backward()
                    optimizer.step()

                    batch_loss += loss.item()
                    tot_loss += loss.item()
                    cnt += 1
                    if cnt >= 1000:
                        #print("loss:", tot_loss / cnt)
                        return tot_loss / cnt
                if batch_loss / length <= 1e-3:
                    #print("loss:", tot_loss / cnt)
                    return batch_loss / length

                
                
    def get_group_new(self, u, context):
        g = set([u])
        for tensor in context:
            tensor = torch.from_numpy(tensor).float().to(device)
            u_pred = self.u_funcs[u](tensor)
            for i in self.users:
                if abs(self.u_funcs[i](tensor) -  u_pred) < self.gamma:
                    #if u_pred>0.0:
                    g.add(i)
        return g

    def recommend(self, u, context, t):
        self.t = t
        g_list = []
        ucb_list = []
        res_list = []
        sample_rs = []
        s_g = []
        
        if self.user_side ==1:
            paras =  self.u_funcs[u].state_dict()
            self.gmodel.load_state_dict(paras)
        g = self.get_group_new(u,  context)
        self.train_meta(g, t)
        for c in context:
            tensor = torch.from_numpy(c).float().to(device)

            res = self.gmodel(tensor)
            self.gmodel.zero_grad()
            res.backward()
            gra = torch.cat([p.grad.flatten().detach() for p in self.gmodel.parameters()])
            #g_list.append(gra)
            
            sigma2 = self.lamdba * self.nu * gra * gra 
            sigma = torch.sqrt(torch.sum(sigma2))
            #sample_r = res1.item() + res.item() + sigma.item()
            sample_r = res.item() + sigma.item()
            sample_rs.append(sample_r)
            ucb_list.append(sigma.item())
            #res_list.append(res.item()+res1.item())
            res_list.append(res.item())
        res_list = np.array(res_list)
        ucb_list = np.array(ucb_list)
        arm = np.argmax(sample_rs)
        #g = s_g[arm]
        return arm, g,res_list, ucb_list
    
    
    def train(self, u, t):
        
        d = self.u_funcs[u].state_dict()
        #print(self.g)
        for k in d.keys():
            d[k] = self.gmodel.state_dict()[k]
        self.u_funcs[u].load_state_dict(d)
        optimizer = optim.Adam(self.u_funcs[u].parameters(), lr=1e-3)
        
        length = len(self.rewards[u])
        index = np.arange(length)
        np.random.shuffle(index)
        cnt = 0
        tot_loss = 0
        while True:
            batch_loss = 0
            for idx in index:
                c = self.contexts[u][idx]
                r = self.rewards[u][idx]
                optimizer.zero_grad()
                loss = (self.u_funcs[u](c.to(device)) - r)**2                

                loss.backward()
                optimizer.step()

                batch_loss += loss.item()
                tot_loss += loss.item()
                cnt += 1
                if cnt >= 1000:
                    #print("loss:", tot_loss / cnt)
                    return tot_loss / cnt
            if batch_loss / length <= 1e-3:
                #print("loss:", tot_loss / cnt)
                return batch_loss / length
            
            
   

