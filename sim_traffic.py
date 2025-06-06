import os, time, random
import numpy as np
from multiprocessing import Pool
from functools import partial
from sys_model import config_UW, config_RF, User, Channel
from utils import formAction
from algos import greedy, roundRobin, opSDMA, QL, DQL, SBFP

def one_iter(cf, mode, UE_list, Act, S, N_repeat, id_case):
    UE = UE_list[id_case]
    Ch = Channel(cf, mode, UE, Act, 0)
    Res = np.zeros((6,2))
    for _ in range(N_repeat):
        Cap = greedy(cf, [Ch,], Act, S)
        Res[0,:] += np.array([Cap.sum(),(Cap.sum()**2)/(cf.N*(Cap**2).sum())])
        Cap = roundRobin(cf, [Ch,], Act, S)
        Res[1,:] += np.array([Cap.sum(),(Cap.sum()**2)/(cf.N*(Cap**2).sum())])
        Cap = opSDMA(cf, [Ch,], [UE,], Act, S)
        Res[2,:] += np.array([Cap.sum(),(Cap.sum()**2)/(cf.N*(Cap**2).sum())])
        Cap, _, _ = SBFP(cf, [Ch,], Act, S, 0, id_case)
        Res[3,:] += np.array([Cap.sum(),(Cap.sum()**2)/(cf.N*(Cap**2).sum())])
        Cap, _ = QL(cf, [Ch,], Act, S, np.nan, id_case)
        Res[4,:] += np.array([Cap.sum(),(Cap.sum()**2)/(cf.N*(Cap**2).sum())])
        Cap = DQL(cf, mode, [Ch,], [UE,], Act, S)
        Res[5,:] += np.array([Cap.sum(),(Cap.sum()**2)/(cf.N*(Cap**2).sum())])
    return Res

if __name__ == '__main__':
    mode = 1 # 0 for UW, 1 for RF
    t_start = time.time()
    random.seed(0)
    if mode == 0:
        cf = config_UW()
    else:
        cf = config_RF()
    Act = formAction(cf.N, cf.N_max)
    S = np.vstack((np.zeros(cf.N),(cf.N_level/2)*np.ones(cf.N))).astype(int)
    Traffic = np.arange(1.5, 5, 0.5)
    N_case, N_repeat, N_worker = 100, 10, 5
    UE_list = []
    for _ in range(N_case):
        UE_list.append(User(cf, False))
    Result = np.zeros((6,2,Traffic.size))
    for k in range(Traffic.size):
        print(f'Traffic = {Traffic[k]}')
        print(N_case*'-')
        for file in os.listdir('./data/weight'):
            os.remove('./data/weight/' + file)
        cf.traffic = Traffic[k]
        for i in range(N_case):
            Lambda = UE_list[i].Lambda
            UE_list[i].Lambda = cf.traffic*Lambda/np.sum(Lambda)
        func = partial(one_iter, cf, mode, UE_list, Act, S, N_repeat)
        m = 0
        while m < N_case:
            N_mp = min(N_worker, N_case-m)
            with Pool(N_mp) as p:
                Res_list = p.map(func, range(m,m+N_mp))
            for Res in Res_list:
                Result[:,:,k] += Res
            m += N_mp
            print(N_mp*'*', end='')
        print(f' time: {(time.time()-t_start)/3600:.2f} hours')
    Result /= N_case*N_repeat
    
    with open('./data/result/traffic.npy', 'wb') as f:
        np.save(f, Traffic)
        np.save(f, Result)
