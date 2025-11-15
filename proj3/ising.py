import numpy as np

# USER INPUT -----------------------------------------------------------------

Lx = 10 # number of x positions
Ly = 20 # number of y positions

nIter = 100000 # number of iterations

J = 1
k = 1
T = 1

# SETUP ----------------------------------------------------------------------

# initial spin, random
s = np.zeros((Ly, Lx, nIter+1))
s[:,:,0] = np.random.randint(-1, 1, (Ly,Lx))
s[:,:,0][s[:,:,0]==0] = 1

beta = 1/(k*T)


