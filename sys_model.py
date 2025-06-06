import math, cmath
import numpy as np
from random import random, gauss
from numpy.linalg import norm
from scipy.stats import poisson

class config_UW:
    def __init__(self):
        self.c = 1500                           # speed of sound
        ## discretization
        self.N_level = 5                        # discretization level
        self.dt = 50e-3                         # channel sample interval
        ## geometry
        self.buoy_loc = np.array([0,0,-5])
        self.depth = 500                        # max water depth
        self.D_range = [100, 1e3]               # user distance to buoy
        ## transmission related
        self.fc = 10e3                          # center freq
        self.B = 5e3                            # bandwidth
        ## array related
        self.M = 24                             # num of array elements
        self.d = 0.5*self.c/self.fc             # array spacing
        ## user related
        self.N = 6                              # num of users
        self.N_max = 3                          # max num of users served
        self.traffic = 2.5                      # expected num of packets
        self.N_b = 4                            # buffer size
        # noise power
        self.N_P = 10**((50-18*np.log10(self.fc/1e3))/10)*self.B
        self.setPower()
        self.P_out = 0.2                        # outage prob
        self.t_tx = 8*1.5e3/20e3                # Tx time (1500 bytes, 20 kbps)
        # slot length
        self.t_slot = round((self.t_tx+self.D_range[1]/self.c)/self.dt)*self.dt
        self.N_dt = int(self.t_slot/self.dt)    # num of samples per slot
        ## simulation related
        self.T = 100                            # num of slots
        self.N_sample = 1000                    # num of samples each layer
        self.delta = 1e4
        self.T_static = self.T
    def setPower(self):
        dis = self.D_range[1]
        SNR_dB = 15
        SNR = 10**(SNR_dB/10)
        a = (10**(absorption(self.fc/1e3)/10))**(1/1000)
        H_eq = math.sqrt(self.M/((dis**1.5)*(a**dis)))
        self.power = self.N_max*SNR*self.N_P/(H_eq**2)

class config_RF:
    def __init__(self):
        self.c = 3e8                            # speed of sound
        ## discretization
        self.N_level = 5                        # discretization level
        self.dt = 2e-5                          # channel sample interval
        ## geometry
        self.buoy_loc = np.array([0,0,-5])
        self.depth = 500                        # max water depth
        self.D_range = [100, 1e3]               # user distance to buoy
        ## transmission related
        self.fc = 2.5e9                         # center freq
        self.B = 5e6                            # bandwidth
        ## array related
        self.M = 24                             # num of array elements
        self.d = 0.5*self.c/self.fc             # array spacing
        ## user related
        self.N = 6                              # num of users
        self.N_max = 3                          # max num of users served
        self.traffic = 2.5                      # expected num of packets
        self.N_b = 4                            # buffer size
        # noise power
        self.N_P = 1.38e-23*300*self.B
        self.setPower()
        self.P_out = 0.2                        # outage prob
        self.t_tx = 8*1.5e3/20e6                # Tx time (1500 bytes, 20 Mbps)
        # slot length
        self.t_slot = round((self.t_tx+self.D_range[1]/self.c)/self.dt)*self.dt
        self.N_dt = int(self.t_slot/self.dt)    # num of samples per slot
        ## simulation related
        self.T = 100                            # num of slots
        self.N_sample = 1000                    # num of samples each layer
        self.delta = 1e7
        self.T_static = self.T
    def setPower(self):
        dis = self.D_range[1]
        SNR_dB = 15
        SNR = 10**(SNR_dB/10)
        PL_dB = 22.7 + 36.7*math.log10(dis) + 26*math.log10(self.fc/1e9)
        H_eq = math.sqrt(self.M/(10**(PL_dB/10)))
        self.power = self.N_max*SNR*self.N_P/(H_eq**2)

class User:
    def __init__(self, cf, iniLoc, fileName=''):
        if iniLoc:
            self.assignLoc(fileName)
        else:
            self.Loc = np.zeros((cf.N,3))
            self.Angle = np.zeros(cf.N)
            for n in range(cf.N):
                dis = 0
                while dis<cf.D_range[0] or dis>cf.D_range[1]:
                    x = cf.D_range[1]*random()
                    y = cf.D_range[1]*(2*random()-1)
                    z = cf.buoy_loc[2]-(cf.buoy_loc[2]+cf.depth)*random()
                    dis = norm(np.array([x,y,z])-cf.buoy_loc)
                self.Loc[n,:] = [x,y,z]
                self.Angle[n] = math.asin((z-cf.buoy_loc[2])/dis)
        while True:
            Lambda = np.array([random() for _ in range(cf.N)])
            if not np.any(Lambda<np.mean(Lambda)/5):
                break
        self.Lambda = cf.traffic*Lambda/np.sum(Lambda)
        self.V = np.zeros((cf.N,3))
    def assignLoc(self, fileName):
        with open(fileName, 'rb') as f:
            Loc = np.load(f)
            Angle = np.load(f)
        self.Loc = Loc
        self.Angle = Angle

class Channel:
    # mode: 0 for UW, 1 for RF
    def __init__(self, cf, mode, UE, Act, loc_err):
        self.H_0 = np.zeros(cf.N)
        self.H2_out = []
        self.BF_real = np.zeros((cf.N,cf.N))
        self.BF_sim = np.zeros((cf.N,cf.N))
        self.Center = np.zeros((cf.N,cf.N_level))
        self.P_b = []
        self.bufferPTM(cf, UE, Act)
        self.P_c = []
        if mode==0:
            self.sim_UW(cf, UE, loc_err)
        else:
            self.sim_RF(cf, UE)
    def bufferPTM(self, cf, UE, Act):
        N_act = Act.shape[0]
        P_arr = np.zeros((cf.N_b+1,cf.N_b+1,cf.N))
        for n in range(cf.N):
            Prob = [poisson.pmf(i,UE.Lambda[n]) for i in range(cf.N_b+1)]
            for i in range(cf.N_b+1):
                P_arr[i,i:,n] = Prob[:cf.N_b-i+1]
                P_arr[i,-1,n] += 1 - P_arr[i,:,n].sum()
        for n in range(cf.N):
            P_b = np.zeros((cf.N_b+1,cf.N_b+1,N_act))
            for a in range(N_act):
                if Act[a,n]==0:
                    P_b[:,:,a] = P_arr[:,:,n]
                else:
                    P_b[1:,:,a] = P_arr[:-1,:,n]
                # store the cumulative probability matrix
                P_b[:,:,a] = np.cumsum(P_b[:,:,a], axis=1)
            self.P_b.append(P_b)
    def sim_UW(self, cf, UE, loc_err):
        f, sig2s, sig2b, B_delp, Sp = cf.fc, 1.125, 0.5625, 1e-4, 5
        mu_p, nu_p = 0.5/Sp, 1e-6
        if loc_err == 0:
            Loc_est, Angle_est = UE.Loc, UE.Angle
        else:
            Loc_est, Angle_est = np.zeros((cf.N,3)), np.zeros(cf.N)
            for n in range(cf.N):
                err_vec = np.array([random()-0.5 for _ in range(3)])
                Loc_est[n,:] = UE.Loc[n,:] + err_vec * loc_err/norm(err_vec)
                Angle_est[n] = math.asin((Loc_est[n,2]-cf.buoy_loc[2])/
                                         norm(Loc_est[n,:]-cf.buoy_loc))
        # Simulator using the estimated location
        N_sample = int(2e4)
        for n in range(cf.N):
            ht, hr = -cf.buoy_loc[2], -Loc_est[n,2]
            d = norm(Loc_est[n,:2]-cf.buoy_loc[:2])
            H_0, tau, theta, ns, nb, hp = mpgeometry(cf.depth, ht, hr, d, f)
            sig_delp = np.sqrt(((2*np.sin(theta))**2)*(ns*sig2s+nb*sig2b))/cf.c
            rho_p = np.exp(-((2*np.pi*f)**2)*(sig_delp**2/2))
            Bp = ((2*math.pi*f*sig_delp)**2)*B_delp
            gamma_bar_p = mu_p + mu_p*Sp*rho_p
            sig_p = np.sqrt(0.5*(mu_p**2*Sp*(1-rho_p**2)+Sp*nu_p**2))
            alpha_p = np.exp(-math.pi*Bp*cf.dt)
            L = len(tau)
            del_gamma = np.zeros(N_sample,dtype=complex)
            gamma = np.zeros((L,N_sample),dtype=complex)
            gamma[:,0] = gamma_bar_p
            H = np.zeros((cf.M,L),dtype=complex)
            phi = math.atan((Loc_est[n,1]-cf.buoy_loc[1])/
                            (Loc_est[n,0]-cf.buoy_loc[0]))
            for l in range(L):
                for i in range(1,N_sample):
                    wp = math.sqrt(sig_p[l]**2*(1-alpha_p[l]**2))* \
                        (gauss(0,1)+1j*gauss(0,1))
                    del_gamma[i] = alpha_p[l]*del_gamma[i-1] + wp
                gamma[l,:] = gamma_bar_p[l] + del_gamma
                path_vec = np.array([math.cos(theta[l])*math.cos(phi), \
                    math.cos(theta[l])*math.sin(phi), math.sin(theta[l])])
                v_l = path_vec@UE.V[n,:]/norm(path_vec)
                H[:,l] = hp[l]*np.exp(-1j*2*math.pi*f*(1-v_l/cf.c)*(tau[l]+ \
                    np.arange(cf.M)*cf.d*math.sin(theta[l])/cf.c))
            for n_bf in range(cf.N):
                W_bf = (1/math.sqrt(cf.M))*np.exp(-1j*2*math.pi*f* \
                    np.arange(cf.M)*cf.d*math.sin(Angle_est[n_bf])/cf.c)
                self.BF_sim[n_bf,n] = np.abs(np.conj(W_bf)@H[:,0])
            W = (1/math.sqrt(cf.M))*np.exp(-1j*2*math.pi*f*np.arange(cf.M)* \
                cf.d*math.sin(Angle_est[n])/cf.c)
            H_eq = np.abs(H_0*np.conj(W)@H@gamma)
            H2, H2_out = H_eq**2, np.zeros(N_sample-cf.N_dt)
            for i in range(H2_out.size):
                H2_out[i] = np.quantile(H2[i:i+cf.N_dt],cf.P_out)
            P_cum = np.arange(1/(cf.N_level+1),0.99,1/(cf.N_level+1))
            Center = np.sqrt(np.quantile(H2_out, P_cum))
            State = np.zeros(H2_out.size,dtype=int)
            P_c = np.zeros((cf.N_level,cf.N_level))
            for i in range(H2_out.size):
                State[i] = np.argmin(np.abs(math.sqrt(H2_out[i])-Center))
                if i>=cf.N_dt:
                    P_c[State[i-cf.N_dt],State[i]] += 1
            # store the cumulative probability matrix
            for j in range(cf.N_level):
                P_c[j,:] /= np.sum(P_c[j,:])
                P_c[j,:] = np.cumsum(P_c[j,:])
            self.H_0[n] = H_0
            self.Center[n,:] = Center
            self.P_c.append(P_c)
        # Real channel
        N_sample = int(cf.N_dt*cf.T_static)
        for n in range(cf.N):
            ht, hr = -cf.buoy_loc[2], -UE.Loc[n,2]
            d = norm(UE.Loc[n,:2]-cf.buoy_loc[:2])
            H_0, tau, theta, ns, nb, hp = mpgeometry(cf.depth, ht, hr, d, f)
            sig_delp = np.sqrt(((2*np.sin(theta))**2)*(ns*sig2s+nb*sig2b))/cf.c
            rho_p = np.exp(-((2*np.pi*f)**2)*(sig_delp**2/2))
            Bp = ((2*math.pi*f*sig_delp)**2)*B_delp
            gamma_bar_p = mu_p + mu_p*Sp*rho_p
            sig_p = np.sqrt(0.5*(mu_p**2*Sp*(1-rho_p**2)+Sp*nu_p**2))
            alpha_p = np.exp(-math.pi*Bp*cf.dt)
            L = len(tau)
            del_gamma = np.zeros(N_sample,dtype=complex)
            gamma = np.zeros((L,N_sample),dtype=complex)
            gamma[:,0] = gamma_bar_p
            H = np.zeros((cf.M,L),dtype=complex)
            phi = math.atan((UE.Loc[n,1]-cf.buoy_loc[1])/
                            (UE.Loc[n,0]-cf.buoy_loc[0]))
            for l in range(L):
                for i in range(1,N_sample):
                    wp = math.sqrt(sig_p[l]**2*(1-alpha_p[l]**2))* \
                        (gauss(0,1)+1j*gauss(0,1))
                    del_gamma[i] = alpha_p[l]*del_gamma[i-1] + wp
                gamma[l,:] = gamma_bar_p[l] + del_gamma
                path_vec = np.array([math.cos(theta[l])*math.cos(phi), \
                    math.cos(theta[l])*math.sin(phi), math.sin(theta[l])])
                v_l = path_vec@UE.V[n,:]/norm(path_vec)
                H[:,l] = hp[l]*np.exp(-1j*2*math.pi*f*(1-v_l/cf.c)*(tau[l]+ \
                    np.arange(cf.M)*cf.d*math.sin(theta[l])/cf.c))
            for n_bf in range(cf.N):
                W_bf = (1/math.sqrt(cf.M))*np.exp(-1j*2*math.pi*f* \
                    np.arange(cf.M)*cf.d*math.sin(Angle_est[n_bf])/cf.c)
                self.BF_real[n_bf,n] = np.abs(np.conj(W_bf)@H[:,0])
            W = (1/math.sqrt(cf.M))*np.exp(-1j*2*math.pi*f*np.arange(cf.M)* \
                cf.d*math.sin(Angle_est[n])/cf.c)
            H_eq = np.abs(H_0*np.conj(W)@H@gamma)
            H2, H2_out = H_eq**2, np.zeros(cf.T_static)
            for i in range(cf.T_static):
                H2_out[i] = np.quantile(H2[i*cf.N_dt:(i+1)*cf.N_dt],cf.P_out)
            self.H2_out.append(H2_out)
    def sim_RF(self, cf, UE):
        f, fD = cf.fc, 50
        N_repeat = int(2e4)
        for n in range(cf.N):
            d = norm(UE.Loc[n,:]-cf.buoy_loc)
            tau = d / cf.c
            phi = math.atan((UE.Loc[n,1]-cf.buoy_loc[1])/
                            (UE.Loc[n,0]-cf.buoy_loc[0]))
            PL = 10**((22.7+36.7*math.log10(d)+26*math.log10(f/1e9))/10)
            H_0 = math.sqrt(1/PL)
            dir_vec = np.array([math.cos(UE.Angle[n])*math.cos(phi), \
                math.cos(UE.Angle[n])*math.sin(phi), math.sin(UE.Angle[n])])
            v_dir = dir_vec @ UE.V[n,:] / norm(dir_vec)
            H = np.zeros((cf.M,N_repeat),dtype=complex)
            M, theta = 16, 2*math.pi*random()
            alpha = [2*math.pi*random() for _ in range(M)]
            beta = [2*math.pi*random() for _ in range(M)]
            for i in range(N_repeat):
                h_I, h_Q = 0, 0
                for m in range(M):
                    h_I += math.cos(2*math.pi*fD*math.cos(((2*m-1)*math.pi+
                        theta)/(4*M))*i*cf.dt + alpha[m])
                    h_Q += math.sin(2*math.pi*fD*math.cos(((2*m-1)*math.pi+
                        theta)/(4*M))*i*cf.dt + beta[m])
                h_I, h_Q = h_I/math.sqrt(M), h_Q/math.sqrt(M)
                H[:,i] = (h_I+1j*h_Q) * np.exp(-1j*2*math.pi*f*(1-v_dir/cf.c)*
                    (tau+np.arange(cf.M)*cf.d*math.sin(UE.Angle[n])/cf.c))
            for n_bf in range(cf.N):
                W_bf = (1/math.sqrt(cf.M))*np.exp(-1j*2*math.pi*f* \
                    np.arange(cf.M)*cf.d*math.sin(UE.Angle[n_bf])/cf.c)
                H_rx = np.exp(-1j*2*math.pi*f*(1-v_dir/cf.c)*(tau+
                    np.arange(cf.M)*cf.d*math.sin(UE.Angle[n])/cf.c))
                self.BF_real[n_bf,n] = np.abs(np.conj(W_bf)@H_rx)
                self.BF_sim[n_bf,n] = np.abs(np.conj(W_bf)@H_rx)
            W = (1/math.sqrt(cf.M))*np.exp(-1j*2*math.pi*f*np.arange(cf.M)*
                cf.d*math.sin(UE.Angle[n])/cf.c)
            H_eq = np.abs(H_0*np.conj(W)@H)
            H2, H2_out = H_eq**2, np.zeros(N_repeat-cf.N_dt)
            for i in range(H2_out.size):
                H2_out[i] = np.quantile(H2[i:i+cf.N_dt],cf.P_out)
            P_cum = np.arange(1/(cf.N_level+1),0.99,1/(cf.N_level+1))
            Center = np.sqrt(np.quantile(H2_out, P_cum))
            State = np.zeros(H2_out.size,dtype=int)
            P_c = np.zeros((cf.N_level,cf.N_level))
            for i in range(H2_out.size):
                State[i] = np.argmin(np.abs(math.sqrt(H2_out[i])-Center))
                if i>=cf.N_dt:
                    P_c[State[i-cf.N_dt],State[i]] += 1
            # store the cumulative probability matrix
            for j in range(cf.N_level):
                P_c[j,:] /= np.sum(P_c[j,:])
                P_c[j,:] = np.cumsum(P_c[j,:])
            self.H_0[n] = H_0
            self.H2_out.append(H2_out[0:cf.N_dt*cf.T_static:cf.N_dt])
            self.Center[n,:] = Center
            self.P_c.append(P_c)

############################ helper functions #################################
def absorption(f):
    return 0.11*f**2/(1+f**2) + 44*f**2/(4100+f**2) + 2.75*1e-4*f**2 + 0.003

def reflCoeff(theta, c1, c2):
    rho1, rho2 = 1000, 1800
    x1 = rho2/c1*np.sin(theta)
    x2 = rho1/c2*np.sqrt(1-(c2/c1)**2*np.cos(theta)**2)
    thetac = cmath.acos(c1/c2).real
    if theta<thetac:
        if thetac==0:
            refl = -1
        else:
            refl = np.exp(1j*np.pi*(1-theta/thetac))
    else:
        refl = (x1-x2)/(x1+x2)
    return refl

def mpgeometry(h, ht, hr, d, f):
    k, c, c2 = 1.5, 1500, 1300
    cut, nr, path = 10, 0, [0,]
    a = (10**(absorption(f/1000)/10))**(1/1000)
    theta, l = [np.arctan((ht-hr)/d),], [np.sqrt((ht-hr)**2+d**2),]
    tau, A = [0,], [(l[0]**k)*(a**l[0]),]
    G, H_0 = [1/np.sqrt(A[0]),], 1/np.sqrt(A[0])
    Gamma, hp, ns, nb = [1,], [1,], [0,], [0,]
    while min(np.abs(G))>=G[0]/cut:
        nr += 1
        first, last = path[0], path[-1]
        nb.append(sum(path))
        ns.append(nr-nb[-1])
        heff = (1-first)*ht+first*(h-ht)+(nr-1)*h+(1-last)*hr+last*(h-hr)
        l.append(np.sqrt(heff**2+d**2))
        theta.append(np.arctan(heff/d))
        if first==1:
            theta[-1] = -theta[-1]
        tau.append((l[-1]-l[0])/c)
        A.append((l[-1]**k)*(a**l[-1]))
        Gamma.append((reflCoeff(abs(theta[-1]),c,c2)**nb[-1])*((-1)**ns[-1]))
        G.append(Gamma[-1]/np.sqrt(A[-1]))
        hp.append(Gamma[-1]/np.sqrt(((l[-1]/l[0])**k)*(a**(l[-1]-l[0]))))
        path = [1-i for i in path]
        first, last = path[0], path[-1]
        nb.append(sum(path))
        ns.append(nr-nb[-1])
        heff = (1-first)*ht+first*(h-ht)+(nr-1)*h+(1-last)*hr+last*(h-hr)
        l.append(np.sqrt(heff**2+d**2))
        theta.append(np.arctan(heff/d))
        if first==1:
            theta[-1] = -theta[-1]
        tau.append((l[-1]-l[0])/c)
        A.append((l[-1]**k)*(a**l[-1]))
        Gamma.append((reflCoeff(abs(theta[-1]),c,c2)**nb[-1])*((-1)**ns[-1]))
        G.append(Gamma[-1]/np.sqrt(A[-1]))
        hp.append(Gamma[-1]/np.sqrt(((l[-1]/l[0])**k)*(a**(l[-1]-l[0]))))
        path.append(1-path[-1])
    return H_0, tau, theta, np.array(ns), np.array(nb), hp
