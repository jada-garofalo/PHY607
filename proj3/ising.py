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
s = np.zeros((nIter+1, Ly, Lx))
s[0,:,:] = np.random.randint(-1, 1, (Ly,Lx))
s[0,:,:][s[0,:,:]==0] = 1

beta = 1/(k*T)

# SOLVER ---------------------------------------------------------------------

for n in range(nIter):
    # fill in next page of spin matrix
    s[n+1,:,:] = s[n,:,:]
   
    # pick a spin site randomly
    i = np.random.randint(0, Lx)
    j = np.random.randint(0, Ly)
   
    # calculate change in energy with the flipped spin 
    # (wrap around lattice boundaries)
    if i == 0:
        sx_sum = s[n,j,i+1] + s[n,j,Lx-1]
    elif i == Lx-1:
        sx_sum = s[n,j,0] + s[n,j,i-1]
    else:
        sx_sum = s[n,j,i+1] + s[n,j,i-1]
    if j == 0:
        sy_sum = s[n,j+1,i] + s[n,Ly-1,i]
    elif j == Ly-1:
        sy_sum = s[n,0,i] + s[n,j-1,i]
    else:
        sy_sum = s[n,j+1,i] + s[n,j-1,i]
       
    dE = 2*s[n,j,i]*J*(sx_sum+sy_sum)
   
    # if the change in energy is not positive, keep the flipped spin
    if dE <= 0:
        s[n+1,j,i] = -s[n,j,i]
    # if the change in energy is positive, keep with probability exp(-beta*dE)
    elif np.random.rand() < np.exp(-beta*dE):
        s[n+1,j,i] = -s[n,j,i]
           
    # repeat
