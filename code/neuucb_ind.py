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

    
    
class neuucb_ind:
    def __init__(self, dim, n, n_arm, lr = 0.01, hidden=100, lamdba = 1.0, nu = 0.01):
        self.context_list = []
        self.reward = []
        self.lr = lr
        self.dim = dim
        self.hidden = hidden
        
        self.lamdba = lamdba
        self.nu = nu
        self.g = []
        
        self.users = range(n)
        self.u_funcs = {}
        self.u_his = {}
        self.U = {}
        self.inip = {}
        for i in range(n):
            self.u_funcs[i] = Network_u(dim, hidden_size=hidden).to(device)
            self.inip[i] = self.u_funcs[i].state_dict()
            total_param = sum(p.numel() for p in self.u_funcs[i].parameters() if p.requires_grad)
            self.U[i] = lamdba * torch.ones((total_param,)).to(device)
            
        
        self.contexts = defaultdict(list)
        self.rewards = defaultdict(list)
        
    
    
    
    def update(self, u, context, reward):
        self.contexts[u].append(torch.from_numpy(context.reshape(1, -1)).float())
        self.rewards[u].append(reward)


    def recommend(self, u, context, t):
        g_list = []
        ucb_list = []
        res_list = []
        sample_rs = []
        for c in context:
            tensor = torch.from_numpy(c).float().to(device)
            res = self.u_funcs[u](tensor)

            self.u_funcs[u].zero_grad()
            res.backward(retain_graph=True)
            gra = torch.cat([p.grad.flatten().detach() for p in self.u_funcs[u].parameters()])
            g_list.append(gra)
                        
            sigma2 = self.lamdba * self.nu * gra * gra / self.U[u]
            sigma = torch.sqrt(torch.sum(sigma2))
            sample_r = res.item() + sigma.item()
            sample_rs.append(sample_r)
            ucb_list.append(sigma.item())
            res_list.append(res.item())
            
        res_list = np.array(res_list)
        ucb_list = np.array(ucb_list)
        arm = np.argmax(sample_rs)
        self.U[u] += g_list[arm] * g_list[arm]
        #g_list = self.gradient_feature(g_list)
        return arm, res_list, ucb_list
    
    
    def train(self, u, t):
        optimizer = optim.SGD(self.u_funcs[u].parameters(), lr=self.lr)
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
                if cnt >= 500:
                    return tot_loss / cnt
            if batch_loss / length <= 1e-3:
                return batch_loss / length

