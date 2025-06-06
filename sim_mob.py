import os, time, random, math
import numpy as np
from multiprocessing import Pool
from functools import partial
from sys_model import config_UW, User, Channel
from utils import formAction, updateModel
from algos import QL, DQL, SBFP

def one_iter(cf, Act, S, UE_ini, Ch_ini, sigma, id_repeat):
    Mob = np.zeros(5)
    Mob[0] = math.pi * (1 + 0.1*random.gauss(0,1))
    Mob[1] = math.pi * (1 + 0.1*random.gauss(0,1))
    Mob[2] = math.pi * (2 + 0.2*random.gauss(0,1))
    Mob[3] = 3 + 0.3*random.gauss(0,1)
    Mob[4] = 1 + 0.1*random.gauss(0,1)
    UE_list, Ch_list = updateModel(cf, UE_ini, Ch_ini, Mob, sigma, Act)
    Res = np.zeros((5,3))
    Cap, t0, _ = SBFP(cf, Ch_list, Act, S, 0, 'm')
    Res[0,:] += np.array([Cap.sum(),(Cap.sum()**2)/(cf.N*(Cap**2).sum()),t0])
    Cap, t1, _ = SBFP(cf, Ch_list, Act, S, 1, 'm')
    Res[1,:] += np.array([Cap.sum(),(Cap.sum()**2)/(cf.N*(Cap**2).sum()),t1])
    Cap, t2, T_re = SBFP(cf, Ch_list, Act, S, 2, 'm')
    Res[2,:] += np.array([Cap.sum(),(Cap.sum()**2)/(cf.N*(Cap**2).sum()),t2])
    N_s_total = cf.N_sample*np.sum(T_re*np.arange(cf.T,0,-1))
    Cap, t3 = QL(cf, Ch_list, Act, S, N_s_total, 'm')
    Res[3,:] += np.array([Cap.sum(),(Cap.sum()**2)/(cf.N*(Cap**2).sum()),t3])
    Cap = DQL(cf, 0, Ch_list, UE_list, Act, S)
    Res[4,:] += np.array([Cap.sum(),(Cap.sum()**2)/(cf.N*(Cap**2).sum()),0])
    return Res, T_re

if __name__ == '__main__':
    t_start = time.time()
    random.seed(0)
    cf = config_UW()
    cf.T_static = 2
    Act = formAction(cf.N, cf.N_max)
    S = np.vstack((np.zeros(cf.N),(cf.N_level/2)*np.ones(cf.N))).astype(int)
    Sigma = np.arange(2, 11, 2)
    N_repeat, N_worker = 50, 5
    Result, Freq_plan = np.zeros((5,3,Sigma.size)), np.zeros((Sigma.size,cf.T))
    for file in os.listdir('./data/weight'):
        os.remove('./data/weight/' + file)
    UE_ini = User(cf, True, './data/loc_ini/two_group.npy')
    Ch_ini = Ch = Channel(cf, 0, UE_ini, Act, 0)
    for k in range(Sigma.size):
        print(f'sigma = {Sigma[k]}')
        print(N_repeat*'-')
        func = partial(one_iter, cf, Act, S, UE_ini, Ch_ini, Sigma[k])
        m = 0
        while m < N_repeat:
            N_mp = min(N_worker, N_repeat-m)
            with Pool(N_mp) as p:
                Res_list = p.map(func, range(m,m+N_mp))
            for Res in Res_list:
                Result[:,:,k] += Res[0]
                Freq_plan[k,:] += Res[1]
            m += N_mp
            print(N_mp*'*',end='')
        print(f' time: {(time.time()-t_start)/3600:.2f} hours')
    Result /= N_repeat
    Freq_plan /= N_repeat
    
    with open('./data/result/mob.npy', 'wb') as f:
        np.save(f, Sigma)
        np.save(f, Result)
        np.save(f, Freq_plan)