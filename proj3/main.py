from isingClass import ising

# USER INPUT -------------------------
Lx = 10 # number of x positions
Ly = 20 # number of y positions
nIter = 100000 # number of iterations
J = 1 # ferromagnetic interaction
k = 1 # boltzmann constant
T = 1 # temperature
# ------------------------------------

ising_2D = ising(Lx, Ly, nIter, J, k, T)
spin = ising_2D.mcmc_hand_written()
Mag = ising_2D.Magnetization(spin)

