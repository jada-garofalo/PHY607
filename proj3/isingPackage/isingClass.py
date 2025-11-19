import numpy as np

class ising:

    def __init__(self, Lx, Ly, nIter, J, k, T):
        """
        Params
        ------
        Lx: number of sites in the x direction
        Ly: number of sites in the y direction
        nIter: number of iterations in the simulation
        J: ferromagnetic interaction, J>0 for ferromagnetic, J<0 for antiferromagnetic
        k: boltzmann constant
        T: temperature
        """
        self.Lx = Lx
        self.Ly = Ly
        self.nIter = nIter
        self.J = J
        self.k = k
        self.T = T
        
    def mcmc_hand_written(self):
        # initial spin, random
        s = np.zeros((self.nIter+1, self.Ly, self.Lx))
        s[0,:,:] = np.random.randint(-1, 1, (self.Ly, self.Lx))
        s[0,:,:][s[0,:,:]==0] = 1

        beta = 1/(self.k*self.T)

        for n in range(self.nIter):
            # fill in next page of spin matrix
            s[n+1,:,:] = s[n,:,:]
           
            # pick a spin site randomly
            i = np.random.randint(0, self.Lx)
            j = np.random.randint(0, self.Ly)
           
            # calculate change in energy with the flipped spin 
            # (wrap around lattice boundaries)
            if i == 0:
                sx_sum = s[n,j,i+1] + s[n,j,self.Lx-1]
            elif i == self.Lx-1:
                sx_sum = s[n,j,0] + s[n,j,i-1]
            else:
                sx_sum = s[n,j,i+1] + s[n,j,i-1]
            if j == 0:
                sy_sum = s[n,j+1,i] + s[n,self.Ly-1,i]
            elif j == self.Ly-1:
                sy_sum = s[n,0,i] + s[n,j-1,i]
            else:
                sy_sum = s[n,j+1,i] + s[n,j-1,i]
               
            dE = 2*s[n,j,i]*self.J*(sx_sum+sy_sum)
           
            # if the change in energy is not positive, keep the flipped spin
            if dE <= 0:
                s[n+1,j,i] = -s[n,j,i]
            # if the change in energy is positive, keep with probability exp(-beta*dE)
            elif np.random.rand() < np.exp(-beta*dE):
                s[n+1,j,i] = -s[n,j,i]
                
        return s
        
    def Magnetization(self, s):
        s = np.asarray(s)
        if s.ndim == 3:
            return np.sum(s, axis=(1, 2))
        elif s.ndim == 2:
            return np.sum(s)
        else:
            raise ValueError("s must be 2D or 3D")

    def energy_config(self, config):
        right = np.roll(config, -1, axis=1)
        down = np.roll(config, -1, axis=0)
        return -self.J * np.sum(config * (right + down))

    def Energy(self, s):
        s = np.asarray(s)
        if s.ndim == 3:
            return np.array([self.energy_config(s[i]) for i in range(s.shape[0])])
        elif s.ndim == 2:
            return self.energy_config(s)
        else:
            raise ValueError("s must be 2D or 3D")

