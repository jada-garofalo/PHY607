import numpy as np
from isingClass import ising
from analysisClass import analysis

# USER INPUT -------------------------
Lx = 10
Ly = 20
nIter = 10000
J = 1
k = 1
T = 1
# ------------------------------------

def gelman_rubin(chains):
    """
    Gelman-Rubin R-hat for a list of 1D chains.
    """
    chains = [np.asarray(c) for c in chains]
    m = len(chains)
    n = min(len(c) for c in chains)
    chains = [c[:n] for c in chains]

    means = np.array([c.mean() for c in chains])
    W = np.mean([np.var(c, ddof=1) for c in chains])
    B = n * np.var(means, ddof=1)

    var_hat = (1 - 1/n) * W + B / n
    return np.sqrt(var_hat / W)


# -------------------------------------------------
# 1. HANDWRITTEN MCMC CHAIN
# -------------------------------------------------

ising_2D = ising(Lx, Ly, nIter, J, k, T)
an = analysis()

chain1 = ising_2D.mcmc_hand_written()
chain2 = ising_2D.mcmc_hand_written()

mag1 = ising_2D.Magnetization(chain1)
mag2 = ising_2D.Magnetization(chain2)

energy1 = ising_2D.Energy(chain1)
energy2 = ising_2D.Energy(chain2)

tau1, _ = an.ACL(mag1)
tau2, _ = an.ACL(mag2)

rhat_mag = gelman_rubin([mag1, mag2])

print("=======================================")
print(" HANDWRITTEN METROPOLIS RESULTS")
print("=======================================")
print(f"ACL Chain 1 = {tau1:.3f}")
print(f"ACL Chain 2 = {tau2:.3f}")
print(f"R-hat (magnetization) = {rhat_mag:.5f}")
print()

# -------------------------------------------------
# 2. PACKAGE COMPARISON SAMPLER
# -------------------------------------------------

import emcee

def log_prior(theta):
    J = theta[0]
    if 0 < J < 5:
        return 0.0
    return -np.inf

def log_likelihood(theta, observed_energy):
    J = theta[0]
    #simple forward model: E = -J * (sum neighbors)
    model_energy = -J * observed_energy
    return -0.5 * ((observed_energy - model_energy)**2) / (0.1**2)

def log_prob(theta, observed_energy):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, observed_energy)

#observed energy from handwritten chain
obs_energy = energy1.mean()

ndim = 1
nwalkers = 20
p0 = 1.0 + 1e-4 * np.random.randn(nwalkers, ndim)

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=[obs_energy])
sampler.run_mcmc(p0, 5000, progress=True)

flat_samples = sampler.get_chain(discard=1000, thin=10, flat=True)
J_samples = flat_samples[:, 0]

rhat_J = gelman_rubin([J_samples[:2000], J_samples[2000:4000]])
tau_J, _ = an.ACL(J_samples)

print("=======================================")
print("   EMCEE PARAMETER-SAMPLER RESULTS")
print("=======================================")
print(f"Mean J = {np.mean(J_samples):.3f}")
print(f"ACL (J) = {tau_J:.3f}")
print(f"R-hat(J) = {rhat_J:.5f}")

