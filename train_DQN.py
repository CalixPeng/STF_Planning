import math
import numpy as np
from random import random, randint, sample, choices
from collections import namedtuple, deque
import torch
import torch.nn as nn
import torch.optim as optim
from sys_model import config_UW, config_RF, User, Channel
from utils import formAction, validAct, step
from algos import DQN, calStateValue_NN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Transition = namedtuple('Transition',
                        ('state', 'act', 'state_n', 'cap', 'loc'))
class ReplayMemory(object):
    def __init__(self, max_num):
        self.memory = deque([], maxlen=max_num)
    
    def push(self, *args):
        self.memory.append(Transition(*args))
    
    def sample(self, size):
        return sample(self.memory, size)
    
    def __len__(self):
        return len(self.memory)

def select_action(Sb, Sc_old, Act, Loc, t, prog_ratio):
    if np.sum(Sb) == 0:
        return 0, Act[0,:]
    eps = 0.2 + 0.8 * math.exp(-prog_ratio)
    if random() > eps:
        id_a, _ = calStateValue_NN(np.hstack((Sb,Sc_old)), Act, Loc.flatten(), 
                                   t, net)
    else:
        id_a = randint(1,Act.shape[0]-1)
        while not validAct(Sb, Act[id_a,:]):
            id_a = randint(1,Act.shape[0]-1)
    return id_a, Act[id_a,:]

def optimize_model(memory, t, Act, batch_size, delta):
    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))
    state = torch.cat(batch.state)
    act = torch.cat(batch.act)
    state_n = torch.cat(batch.state_n)
    cap = torch.cat(batch.cap)
    loc = torch.cat(batch.loc)
    t_stamp = t*torch.ones(batch_size, 1, device=device)
    net_in = torch.cat((state, act, loc, t_stamp), 1)
    Q_values = net(net_in)
    if t == cf.T-1:
        Q_target = torch.log(cap+delta)
    else:
        Q_target = torch.zeros(Q_values.shape, device=device)
        for i in range(batch_size):
            _, U = calStateValue_NN(state_n[i,:].cpu().numpy(), Act, 
                                    loc[i,:].cpu().numpy(), t+1, net)
            Q_target[i,:] = U + cap[i,:]/torch.exp(U)
            if torch.isnan(Q_target[i,:]).any():
                return -1
    loss = criterion(Q_values, Q_target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

mode = 1

if mode == 0:
    cf = config_UW()
else:
    cf = config_RF()
Act = formAction(cf.N, cf.N_max)

net = DQN(cf.N).to(device)

batch_size, N_epoch, learning_rate = 100, 1000, 5e-4
optimizer = optim.AdamW(net.parameters(), lr=learning_rate, amsgrad=True)
criterion = nn.SmoothL1Loss()

memory_size, sample_size = 200*batch_size, 10*batch_size
memory = ReplayMemory(memory_size)

epoch = 0
while epoch < N_epoch:
    if memory.__len__() > 0 and random() < 0.8:
        print(f'Epoch {epoch}')
        with torch.enable_grad():
            for t in range(cf.T-1, -1, -1):
                loss = optimize_model(memory, t, Act, batch_size, cf.delta)
                if loss > 0:
                    print(f'loss = {loss:.3f}, t = {t}')
                if loss < 1e-2 or loss > 0.5: 
                    break
        epoch += 1
        continue
    UE = User(cf, False)
    Ch = Channel(cf, mode, UE, Act, 0)
    t = 0
    for _ in range(sample_size):
        Sb = np.array([randint(0,cf.N_b) for _ in range(cf.N)])
        Sc_old = np.array([randint(0,cf.N_level-1) for _ in range(cf.N)])
        id_a, act = select_action(Sb, Sc_old, Act, UE.Loc, t, epoch/N_epoch)
        Sc = np.zeros(cf.N, dtype=int)
        for n in range(cf.N):
            Sc[n] = choices(range(cf.N_level),weights=Ch.P_c[n][Sc_old[n],:])[0]
        Cap, S_n = step(cf, Ch, np.vstack((Sb,Sc)), act, id_a, False, np.nan)
        memory.push(torch.tensor(np.hstack((Sb,Sc_old)),dtype=torch.float32,device=device).unsqueeze(0),
                    torch.tensor(Act[id_a,:],dtype=torch.float32,device=device).unsqueeze(0),
                    torch.tensor(np.hstack((S_n[0,:],Sc)),dtype=torch.float32,device=device).unsqueeze(0),
                    torch.tensor(Cap,dtype=torch.float32,device=device).unsqueeze(0),
                    torch.tensor(UE.Loc.flatten(),dtype=torch.float32,device=device).unsqueeze(0))
        t = (t + 1) % cf.T
    print(f'Data collected, {int(len(memory)/batch_size)}/{int(memory_size/batch_size)}')

torch.save(net.state_dict(), './data/dqn.pth')
