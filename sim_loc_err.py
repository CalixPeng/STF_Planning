import os, time, random
import numpy as np
from multiprocessing import Pool
from functools import partial
from sys_model import config_UW, config_RF, User, Channel
from utils import formAction
from algos import greedy, opSDMA, SBFP

def one_case(cf, mode, UE_list, Act, S, loc_err, N_repeat, id_case):
    UE = UE_list[id_case]
    Ch = Channel(cf, mode, UE, Act, loc_err)
    Res = np.zeros((3,2))
    for _ in range(N_repeat):
        Cap = greedy(cf, [Ch,], Act, S)
        Res[0,:] += np.array([Cap.sum(),(Cap.sum()**2)/(cf.N*(Cap**2).sum())])
        Cap = opSDMA(cf, [Ch,], [UE,], Act, S)
        Res[1,:] += np.array([Cap.sum(),(Cap.sum()**2)/(cf.N*(Cap**2).sum())])
        Cap, _, _ = SBFP(cf, [Ch,], Act, S, 0, id_case)
        Res[2,:] += np.array([Cap.sum(),(Cap.sum()**2)/(cf.N*(Cap**2).sum())])
    return Res

if __name__ == '__main__':
    mode = 0 # 0 for UW, 1 for RF
    t_start = time.time()
    random.seed(0)
    if mode == 0:
        cf = config_UW()
    else:
        cf = config_RF()
    Act = formAction(cf.N, cf.N_max)
    S = np.vstack((np.zeros(cf.N),(cf.N_level/2)*np.ones(cf.N))).astype(int)
    Error = np.arange(0, 51, 10)
    N_case, N_repeat, N_worker = 100, 10, 5
    UE_list = []
    for _ in range(N_case):
        UE_list.append(User(cf, False))
    Result = np.zeros((3,2,Error.size))
    for k in range(Error.size):
        print(f'Error = {Error[k]}')
        print(N_case*'-')
        for file in os.listdir('./data/weight'):
            os.remove('./data/weight/' + file)
        func = partial(one_case, cf, mode, UE_list, Act, S, Error[k], N_repeat)
        m = 0
        while m < N_case:
            N_mp = min(N_worker, N_case-m)
            with Pool(N_mp) as p:
                Res_list = p.map(func, range(m,m+N_mp))
            for Res in Res_list:
                Result[:,:,k] += Res
            m += N_mp
            print(N_mp*'*',end='')
        print(f' time: {(time.time()-t_start)/3600:.2f} hours')
    Result /= N_case*N_repeat
    
    with open('./data/result/loc_err.npy', 'wb') as f:
        np.save(f, Error)
        np.save(f, Result)
