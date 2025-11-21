import numpy as np
from .isingClass import ising
from .analysisClass import analysis
import matplotlib.pyplot as plt

# USER INPUT -------------------------
Lx = 10
Ly = 10
nIter = 50000
J_arr = [0.5, 1] # various J to test
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

for J in J_arr:
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
    print(f"J = {J}")
    print(f"ACL Chain 1 = {tau1:.3f}")
    print(f"ACL Chain 2 = {tau2:.3f}")
    print(f"R-hat (magnetization) = {rhat_mag:.5f}")
    print()

    # plots -------------------------------------------
    fig1 = plt.figure(1)
    ax1 = fig1.add_subplot(1, 1, 1)
    fig2 = plt.figure(2)
    ax2 = fig2.add_subplot(1, 1, 1)
    fig3 = plt.figure(3)
    ax3 = fig3.add_subplot(1, 1, 1)
    fig4 = plt.figure(4)
    ax4 = fig4.add_subplot(1, 1, 1)
    for n in range(10):
        s_n = ising_2D.mcmc_hand_written()
        mag_n = ising_2D.Magnetization(s_n)
        E_n = ising_2D.Energy(s_n)
        _, rho_mag_n = an.ACL(mag_n)
        _, rho_E_n = an.ACL(E_n)
        ax1.plot(np.linspace(0,len(mag_n),len(mag_n)),mag_n)
        ax2.plot(np.linspace(0,len(E_n),len(E_n)),E_n)
        ax3.plot(np.linspace(0,len(rho_mag_n),len(rho_mag_n)),rho_mag_n)
        ax4.plot(np.linspace(0,len(rho_E_n),len(rho_E_n)),rho_E_n)
    ax1.set_title(f"Trace plot of magnetization for J={J}")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Magnetization")
    ax2.set_title(f"Trace plot of energy for J={J}")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Energy")
    ax3.set_title(f"ACF of magnetization for J={J}")
    ax3.set_xlabel("Lag")
    ax3.set_ylabel("Autocorrelation")
    ax4.set_title(f"ACF of energy for J={J}")
    ax4.set_xlabel("Lag")
    ax4.set_ylabel("Autocorrelation")
    plt.show()
    if J < 1:
        k = int(10 * J) # file names no decimals
    else:
        k = J
    fig1.savefig(f"trace_magnetization_J{k}.png")
    fig2.savefig(f"trace_energy_J{k}.png")
    fig3.savefig(f"acf_magnetization_J{k}.png")
    fig4.savefig(f"acf_energy_J{k}.png")
    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)
    plt.close(fig4)
J = 1

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

print("END PROJECT SCRIPT")

