import math
import numpy as np
from random import random

def iniLoc(cf):
    Angle = -(math.pi/180) * np.array([7, 10, 12, 78, 80, 83])
    Loc = np.zeros((cf.N, 3))
    for n in range(cf.N):
        dis = 0
        while dis < 2*cf.D_range[0] or dis > cf.D_range[1]:
            z = cf.buoy_loc[2] - (cf.buoy_loc[2]+cf.depth)*random()
            rho = abs((z-cf.buoy_loc[2])/math.tan(Angle[n]))
            x = math.sqrt((rho**2)*random())
            y = math.sqrt((rho**2)-(x**2))
            Loc[n,:] = np.array([x,y,z])
            dis = np.linalg.norm(Loc[n,:]-cf.buoy_loc)
    with open('./data/loc_ini/two_group.npy', 'wb') as f:
        np.save(f, Loc)
        np.save(f, Angle)
    return Loc, Angle
