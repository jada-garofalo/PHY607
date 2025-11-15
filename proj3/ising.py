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

# SOLVER ---------------------------------------------------------------------

for n in range(nIter):
    # fill in next page of spin matrix
    s[:,:,n+1] = s[:,:,n]
   
    # pick a spin site randomly
    i = np.random.randint(0, Lx)
    j = np.random.randint(0, Ly)
   
    # calculate change in energy with the flipped spin 
    # (wrap around lattice boundaries)
    if i == 0:
        sx_sum = s[j,i+1,n] + s[j,Lx-1,n]
    elif i == Lx-1:
        sx_sum = s[j,0,n] + s[j,i-1,n]
    else:
        sx_sum = s[j,i+1,n] + s[j,i-1,n]
    if j == 0:
        sy_sum = s[j+1,i,n] + s[Ly-1,i,n]
    elif j == Ly-1:
        sy_sum = s[0,i,n] + s[j-1,i,n]
    else:
        sy_sum = s[j+1,i,n] + s[j-1,i,n]
       
    dE = 2*s[j,i,n]*J*(sx_sum+sy_sum)
   
    # if the change in energy is not positive, keep the flipped spin
    if dE <= 0:
        s[j,i,n+1] = -s[j,i,n]
    # if the change in energy is positive, keep with probability exp(-beta*dE)
    elif np.random.rand() < np.exp(-beta*dE):
        s[j,i,n+1] = -s[j,i,n]
           
    # repeat
