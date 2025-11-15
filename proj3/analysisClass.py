import numpy as np

class analysis:

    def __init__(self):
        
    def ACL(self, chain, max_lag=None):
        n = len(chain)
        if max_lag is None:
            max_lag = n // 10
        y = chain - np.mean(chain)
        c = np.correlate(y, y, mode='full')
        c = c[n-1 : n-1+max_lag]
        rho = c / c[0]
        negative_indices = np.where(rho < 0)[0]
        if len(negative_indices) > 0:
            cutoff = negative_indices[0]
        else:
            cutoff = len(rho)
        tau = 1.0 + 2.0 * np.sum(rho[1:cutoff])
        return tau, rho
