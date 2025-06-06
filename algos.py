import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from random import sample, randint
from os.path import isfile
from utils import validAct, randSample, step, calCap

def greedy(cf, Ch_list, Act, S):
    S_old, id_a_old = S, 0
    _, S = step(cf, Ch_list[0], S, np.zeros(cf.N), 0, True, 0)
    Result = np.zeros((cf.N,cf.T))
    for t in range(cf.T):
        id_m = int(t/cf.T_static)
        C_max, id_max = 0, 0
        _, S_pred = step(cf, Ch_list[id_m], S_old, Act[id_a_old,:], \
            id_a_old, False, np.nan)
        S_pred[0,:] = S[0,:]
        for a in range(1,Act.shape[0]):
            if not validAct(S[0,:], Act[a,:]):
                continue
            C, _ = step(cf, Ch_list[id_m], S_pred, Act[a,:], a, False, np.nan)
            if C.sum()>C_max:
                C_max = C.sum()
                id_max = a
        id_a = id_max
        S_old, id_a_old = S, id_a
        t_m = t - id_m*cf.T_static
        Cap, S = step(cf, Ch_list[id_m], S, Act[id_a,:], id_a, True, t_m)
        Result[:,t] = Cap
    return np.mean(Result,axis=1)

def roundRobin(cf, Ch_list, Act, S):
    _, S = step(cf, Ch_list[0], S, np.zeros(cf.N), 0, True, 0)
    Result = np.zeros((cf.N,cf.T))
    id_user = 0
    for t in range(cf.T):
        id_m = int(t/cf.T_static)
        if S[0,:].sum()==0:
            id_a = 0
        else:
            while S[0,id_user] == 0:
                id_user = (id_user+1) % cf.N
            act = np.zeros(cf.N)
            act[id_user] = 1
            id_a = np.where((Act==act).all(axis=1))[0][0]
        t_m = t - id_m*cf.T_static
        Cap, S = step(cf, Ch_list[id_m], S, Act[id_a,:], id_a, True, t_m)
        Result[:,t] = Cap
        id_user = (id_user+1) % cf.N
    return np.mean(Result,axis=1)

def opSDMA(cf, Ch_list, UE_list, Act, S):
    _, S = step(cf, Ch_list[0], S, np.zeros(cf.N), 0, True, 0)
    Result = np.zeros((cf.N,cf.T))
    for t in range(cf.T):
        id_m = int(t/cf.T_static)
        if t==0 or id_m!=int((t-1)/cf.T_static):
            Group = groupUser(cf, UE_list[id_m].Angle)
        if max(Group)>cf.N_max:
            Id_g = sample(range(1,Group.max()+1),cf.N_max)
        else:
            Id_g = range(1,Group.max()+1)
        act = np.zeros(cf.N)
        for id_g in Id_g:
            Id_u = np.argwhere((Group==id_g) & (S[0,:]>0)).flatten()
            if Id_u.size>0:
                id_u = sample(list(Id_u),1)[0]
                act[id_u] = 1
        id_a = np.where((Act==act).all(axis=1))[0][0]
        t_m = t - id_m*cf.T_static
        Cap, S = step(cf, Ch_list[id_m], S, Act[id_a,:], id_a, True, t_m)
        Result[:,t] = Cap
    return np.mean(Result,axis=1)

def QL(cf, Ch_list, Act, S, N_s_total, id_case):
    W, t_tr = np.nan, 0
    S_old, Result = S, np.zeros((cf.N,cf.T))
    _, S = step(cf, Ch_list[0], S, np.zeros(cf.N), 0, True, 0)
    for t in range(cf.T):
        id_m = int(t/cf.T_static)
        if t % cf.T_static == 0:
            fName = './data/weight/QL_' + str(id_case) + '.npy'
            if t == 0 and isfile(fName):
                with open(fName, 'rb') as f:
                    W = np.load(f)
                    t_ini = np.load(f)
                t_tr += t_ini
            else:
                t_start = time.time()
                W = updateW_QL(cf, Ch_list[id_m], Act, t, N_s_total, W)
                t_tr += (time.time() - t_start)
                if t == 0 and not isfile(fName):
                    with open(fName, 'wb') as f:
                        np.save(f, W)
                        np.save(f, t_tr)
        _, id_a = calStateValue(cf, Act, S[0,:], S_old[1,:], W[:,:,t])
        S_old, t_m = S, t - id_m*cf.T_static
        Cap, S = step(cf, Ch_list[id_m], S, Act[id_a,:], id_a, True, t_m)
        Result[:,t] = Cap
    return np.mean(Result,axis=1), t_tr/cf.T

def DQL(cf, mode, Ch_list, UE_list, Act, S):
    if mode == 0:
        fileName = './data/dqn_UW.pth'
    else:
        fileName = './data/dqn_RF.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = DQN(cf.N).to(device)
    net.load_state_dict(torch.load(fileName))
    net.eval()
    S_old, Result = S, np.zeros((cf.N,cf.T))
    _, S = step(cf, Ch_list[0], S, np.zeros(cf.N), 0, True, 0)
    for t in range(cf.T):
        id_m = int(t/cf.T_static)
        id_a, _ = calStateValue_NN(np.hstack((S[0,:],S_old[1,:])), Act, 
                                   UE_list[id_m].Loc.flatten(), t, net)
        S_old, t_m = S, t - id_m*cf.T_static
        Cap, S = step(cf, Ch_list[id_m], S, Act[id_a,:], id_a, True, t_m)
        Result[:,t] = Cap
    return np.mean(Result,axis=1)

def SBFP(cf, Ch_list, Act, S, replan, id_case):
    t_tr, T_re = 0, np.zeros(cf.T)
    S_old, Result = S, np.zeros((cf.N,cf.T))
    _, S = step(cf, Ch_list[0], S, np.zeros(cf.N), 0, True, 0)
    W, id_w = np.nan, np.nan
    for t in range(cf.T):
        id_m = int(t/cf.T_static)
        if planRule(cf, replan, t, S, id_m, W, id_w, Act, Ch_list, 
                    np.sum(Result[:,:t],axis=1)):
            fName = './data/weight/SBFP_' + str(id_case) + '.npy'
            if t == 0 and isfile(fName):
                with open(fName, 'rb') as f:
                    W = np.load(f)
                    t_ini = np.load(f)
                t_tr += t_ini
            else:
                t_start = time.time()
                W = linApprox(cf, Ch_list[id_m], Act, t, np.sum(Result[:,:t],axis=1))
                t_tr += (time.time() - t_start)
                if t == 0 and not isfile(fName):
                    with open(fName, 'wb') as f:
                        np.save(f, W)
                        np.save(f, t_tr)
                else:
                    T_re[t] = 1
            id_w = int(t/cf.T_static)
        _, id_a = calStateValue(cf, Act, S[0,:], S_old[1,:], W[:,:,t])
        S_old, t_m = S, t - id_m*cf.T_static
        Cap, S = step(cf, Ch_list[id_m], S, Act[id_a,:], id_a, True, t_m)
        Result[:,t] = Cap
    return np.mean(Result,axis=1), t_tr/cf.T, T_re

############################ helper functions #################################
def groupUser(cf, Angle):
    Group = np.zeros(cf.N, dtype=int)
    Order = np.argsort(Angle)
    Angle_sort_d = Angle[Order]*180/np.pi
    while np.any(Group==0):
        id_g = Group.max() + 1
        for n in range(cf.N):
            if Group[Order[n]]==0:
                id_u, Group[Order[n]] = n, id_g
                break
        for n in range(id_u+1, cf.N):
            if Angle_sort_d[n]-Angle_sort_d[id_u]<20:
                Group[Order[n]] = id_g
            else:
                break
    return Group

def updateW_QL(cf, Ch, Act, t, N_s_total, W):
    N_a, N_f = Act.shape[0], 2*cf.N + Act.shape[0]
    if t == 0:
        batch_sz = int(cf.N_sample/5)
        N_iter = int(cf.N_sample/batch_sz)
        W = np.zeros((cf.N,N_f,cf.T))
    else:
        N_s_once = N_s_total / np.arange(cf.T,0,-2).sum()
        batch_sz = max(100, int(N_s_once/3))
        N_iter = round(N_s_once/batch_sz)
    for i in range(N_iter):
        for tau in range(cf.T-1,t-1,-1):
            F, Goal = np.zeros((batch_sz,N_f)), np.zeros((cf.N,batch_sz))
            W_batch = np.zeros((cf.N,N_f))
            for j in range(batch_sz):         
                Sb = np.array([randint(0,cf.N_b) for _ in range(cf.N)])
                Sc_old = np.array([randint(0,cf.N_level-1) for _ in range(cf.N)])
                id_a = randint(0,N_a-1)
                while not validAct(Sb, Act[id_a,:]):
                    id_a = randint(0,N_a-1)
                Sc = np.zeros(cf.N, dtype=int)
                for n in range(cf.N):
                    Sc[n] = randSample(Ch.P_c[n][Sc_old[n],:])
                Cap, S_n = step(cf, Ch, np.vstack((Sb,Sc)), Act[id_a,:], id_a, False)
                F[j,:cf.N], F[j,cf.N:2*cf.N], F[j,2*cf.N+id_a] = Sb, Sc_old, 1
                if tau == cf.T-1:
                    Goal[:,j] = np.log(Cap+cf.delta)
                else:
                    V, _ = calStateValue(cf, Act, S_n[0,:], Sc, W[:,:,tau+1])
                    Goal[:,j] = V + Cap/np.exp(V)
            for n in range(cf.N):
                W_batch[n,:] = np.linalg.lstsq(F, Goal[n,:], rcond=None)[0]
            if t == 0 and i == 0:
                alpha = 1
            else:
                alpha = batch_sz/cf.N_sample
            W[:,:,tau] += alpha*(W_batch-W[:,:,tau])
    return W

class DQN(nn.Module):
    def __init__(self, N):
        super(DQN, self).__init__()
        N_in, N_out = 2*N + N + 3*N + 1, N
        self.layer1 = nn.Linear(N_in, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, 64)
        self.layer4 = nn.Linear(64, 64)
        self.layer5 = nn.Linear(64, 32)
        self.layer6 = nn.Linear(32, N_out)
    
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        x = F.relu(self.layer5(x))
        return self.layer6(x)

def calStateValue_NN(State, Act, Loc, t, net):
    N_a, N = Act.shape
    with torch.no_grad():
        id_a, U, U_sum_max = 0, torch.zeros(N), 0
        for a in range(N_a):
            if not validAct(State[:N], Act[a,:]):
                continue
            net_in = np.hstack((State, Act[a,:], Loc, t))
            net_in = torch.tensor(net_in, dtype=torch.float32, 
                                  device='cuda').unsqueeze(0)
            U = net(net_in)
            if torch.sum(U) > U_sum_max:
                id_a, U_sum_max = a, torch.sum(U)
    return id_a, U

def planRule(cf, replan, t, S, id_m, W, id_w, Act, Ch_list, Cap_pre):
# replan: no replan (0), always replan (1), replan when necessary (2)
    if t % cf.T_static != 0:
        return False
    if replan == 0:
        return (t == 0)
    elif replan == 1:
        return True
    else:
        if t == 0:
            return True
        C_dif = np.zeros(cf.N)
        for n in range(cf.N):
            for act in Act:
                if not act[n]:
                    continue
                c1_worst = calCap(cf, Ch_list[id_w], n, act, False, np.nan, 0)
                c1_best = calCap(cf, Ch_list[id_w], n, act, False, np.nan, 
                                 cf.N_level-1)
                c2_worst = calCap(cf, Ch_list[id_m], n, act, False, np.nan, 0)
                c2_best = calCap(cf, Ch_list[id_m], n, act, False, np.nan,
                                 cf.N_level-1)
                C_dif[n] = max(C_dif[n], abs(c1_worst-c2_worst), 
                               abs(c1_best-c2_best))
        T_last = max(cf.T-10, t)
        dif = 0
        for tau in range(t+1, T_last):
            V, _ = calStateValue(cf, Act, np.zeros(cf.N), np.zeros(cf.N), 
                                 W[:,:,tau])
            dif += np.sum(C_dif/np.exp(V))
        W1 = linApprox(cf, Ch_list[id_w], Act, T_last, Cap_pre)
        V1, _ = calStateValue(cf, Act, np.zeros(cf.N), np.zeros(cf.N), 
                           W1[:,:,T_last])
        W2 = linApprox(cf, Ch_list[id_m], Act, T_last, Cap_pre)
        V2, _ = calStateValue(cf, Act, np.zeros(cf.N), np.zeros(cf.N), 
                           W2[:,:,T_last])
        dif += np.sum(np.abs(V1-V2))
        V_now, _ = calStateValue(cf, Act, S[0,:], S[1,:], W[:,:,t])
        return (dif/np.sum(V_now) > 0.05)

def calStateValue(cf, Act, Sb, Sc_old, W):
    U_s = W[:,:cf.N] @ Sb + W[:,cf.N:2*cf.N] @ Sc_old
    id_a, U_a_max = 0, 0
    for a in range(Act.shape[0]):
        if not validAct(Sb, Act[a,:]):
            continue
        U_a = np.sum(W[:,2*cf.N+a])
        if U_a > U_a_max:
            id_a, U_a_max = a, U_a
    return U_s+W[:,2*cf.N+id_a], id_a

def linApprox(cf, Ch, Act, t, Cap_pre):
    N_a = Act.shape[0]
    N_f = 2*cf.N + N_a
    delta = max(cf.delta-np.min(Cap_pre),0)
    W = np.zeros((cf.N, N_f, cf.T))
    for tau in range(cf.T-1,t-1,-1):
        F, Goal = np.zeros((cf.N_sample,N_f)), np.zeros((cf.N,cf.N_sample))
        for i in range(cf.N_sample):
            Sb = np.array([randint(0,cf.N_b) for _ in range(cf.N)])
            Sc_old = np.array([randint(0,cf.N_level-1) for _ in range(cf.N)])
            id_a = randint(0,N_a-1)
            while not validAct(Sb, Act[id_a,:]):
                id_a = randint(0,N_a-1)
            F[i,:cf.N], F[i,cf.N:2*cf.N], F[i,2*cf.N+id_a] = Sb, Sc_old, 1
            Sc = np.array([randSample(Ch.P_c[n][Sc_old[n],:]) for n in 
                           range(cf.N)], dtype=int)
            Cap, S_n = step(cf, Ch, np.vstack((Sb,Sc)), Act[id_a,:], id_a, False)
            if tau == cf.T-1:
                Goal[:,i] = np.log(Cap+Cap_pre+delta)
            else:
                U, _ = calStateValue(cf, Act, S_n[0,:], Sc, W[:,:,tau+1])
                Goal[:,i] = U + Cap/np.exp(U)
        if delta > 0 and np.min(np.exp(Goal)) > 2*delta:
            Goal = np.log(np.exp(Goal)-delta)
            delta = 0
        for n in range(cf.N):
            W[n,:,tau] = np.linalg.lstsq(F, Goal[n,:], rcond=None)[0]
    return W
