import math
import numpy as np
from random import random, gauss
from copy import deepcopy
from sys_model import Channel

def formAction(N, N_max):
    N_act = 0
    for i in range(N_max+1):
        N_act += math.comb(N,i)
    Act = np.zeros((N_act,N),dtype=bool)
    count = 0
    for num in range(2**N):
        act = [int(j) for j in bin(num)[2:].zfill(N)]
        if sum(act)>N_max:
            continue
        Act[count,:] = act
        count += 1
    return Act

def validAct(Sb, act):
    for n in range(Sb.size):
        if act[n] and Sb[n]==0:
            return False
    return True

def randSample(P_cum):
    num = random()
    i = 0
    while num > P_cum[i]:
        i += 1
    return i

def step(cf, Ch, S, act, id_act, real, t_m=0):
    Cap, S_n = np.zeros(cf.N), np.zeros((2,cf.N),dtype=int)
    for n in range(cf.N):
        if act[n] != 0:
            Cap[n] = calCap(cf, Ch, n, act, real, t_m, S[1,n])
        S_n[0,n] = randSample(Ch.P_b[n][S[0,n],:,id_act])
        if real:
            H_out = math.sqrt(Ch.H2_out[n][t_m])
            S_n[1,n] = np.argmin(np.abs(H_out-Ch.Center[n,:]))
        else:
            S_n[1,n] = randSample(Ch.P_c[n][S[1,n],:])
    return Cap, S_n

def calCap(cf, Ch, id_rx, act, real, t_m, sc):
    Denom = (1/Ch.H_0)**2 * act
    Power = (cf.power/Denom.sum()) * Denom
    if real:
        H_out = math.sqrt(Ch.H2_out[id_rx][t_m])
        BF_gain = Ch.BF_real
    else:
        H_out = Ch.Center[id_rx,int(sc)]
        BF_gain = Ch.BF_sim
    Sig = np.zeros(cf.N)
    for id_tx in np.where(act!=0)[0]:
        H = H_out * BF_gain[id_tx,id_rx]/BF_gain[id_rx,id_rx]
        Sig[id_tx] = Power[id_tx] * (H**2)
    SINR_out = Sig[id_rx]/(np.sum(Sig)-Sig[id_rx]+cf.N_P)
    cap = cf.B*math.log2(1+SINR_out)*cf.P_out
    return cap

def updateModel(cf, UE_ini, Ch_ini, Mob, sigma, Act):
    UE_list, Ch_list = [UE_ini, ], [Ch_ini, ]
    for id_model in range(math.ceil(cf.T/cf.T_static)):
        UE_list.append(mobilityModel(cf, UE_list[-1], Mob, sigma, id_model))
        Ch_list.append(Channel(cf, 0, UE_list[-1], Act, 0))
    return UE_list, Ch_list

############################ helper functions #################################
def mobilityModel(cf, UE, Mob, sigma, id_model):
    k1, k2, k3, lamda, v = Mob[0], Mob[1], Mob[2], Mob[3], Mob[4]
    t = id_model * cf.T_static * cf.t_slot
    UE = deepcopy(UE)
    for n in range(cf.N):
        Vx = k1*lamda*v*math.sin(k2*UE.Loc[n,0])*math.cos(k3*UE.Loc[n,1]) + \
            k1*lamda*math.cos(2*k1*t) + sigma*gauss(0,1)
        Vy = -lamda*v*math.cos(k2*UE.Loc[n,0])*math.sin(k3*UE.Loc[n,1]) + \
            sigma*gauss(0,1)
        Vz = sigma*gauss(0,1)
        UE.Loc[n,0] = abs(UE.Loc[n,0] + Vx*cf.T_static*cf.t_slot)
        UE.Loc[n,1] = UE.Loc[n,1] + Vy*cf.T_static*cf.t_slot
        UE.Loc[n,2] = UE.Loc[n,2] + Vz*cf.T_static*cf.t_slot
        UE.V[n,:] = [Vx, Vy, Vz]
        if UE.Loc[n,2] > 0:
            UE.Loc[n,2] = -UE.Loc[n,2]
        elif UE.Loc[n,2] < -cf.depth:
            UE.Loc[n,2] = -UE.Loc[n,2]-2*cf.depth
        UE.Angle[n] = math.asin((UE.Loc[n,2]-cf.buoy_loc[2])/
                           np.linalg.norm(UE.Loc[n,:]-cf.buoy_loc))
    return UE
