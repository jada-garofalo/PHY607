import numpy as np
import matplotlib.pyplot as plt
import h5py

f = h5py.File('./data.hdf', 'r')
xpos = f['data/xpos'][:]
ypos = f['data/ypos'][:]
f.close()
plt.scatter(xpos, ypos)
plt.title("Observed Data")
plt.show()

def fifth(params, x):
    zed, un, deux, trois, quatre, cinq = params
    return cinq*(x**5) + quatre*(x**4) + trois*(x**3) + deux*(x**2) + un*x + zed

def log_likelihood(params):
    y_pred = fifth(params, xpos)
    residuals = ypos - y_pred
    return -0.5 * np.sum(residuals**2)

def log_prior(params):
    if np.all(np.abs(params) < 1000):  #restrict values
        return -0.5 * np.sum((params / 10.0)**2)
    return -np.inf  #invalid region

def log_posterior(params):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params)

def stretch_move(walkers, log_probs, a=2.0): #affine invariant ensemble MCMC
    nwalkers, ndim = walkers.shape
    new_walkers = walkers.copy()
    new_log_probs = log_probs.copy()

    for i in range(nwalkers):
        #pick a comp walker
        j = np.random.randint(nwalkers)
        while j == i:
            j = np.random.randint(nwalkers)
        x_j = walkers[j]

        #stretching factor
        z = ((a - 1) * np.random.rand() + 1) ** 2 / a

        #propose new position
        x_prop = x_j + z * (walkers[i] - x_j)
        log_prob_prop = log_posterior(x_prop)

        #acceptance probability
        log_accept = (ndim - 1) * np.log(z) + log_prob_prop - log_probs[i]
        if np.log(np.random.rand()) < log_accept:
            new_walkers[i] = x_prop
            new_log_probs[i] = log_prob_prop

    return new_walkers, new_log_probs

ndim = 6  #6 polynomial coefficients
nwalkers = 500
nsteps = 1000

#random Gaussian initialization
walkers = np.random.randn(nwalkers, ndim)
log_probs = np.array([log_posterior(w) for w in walkers])

walker_history = []

#run MCMC
for step in range(nsteps):
    walkers, log_probs = stretch_move(walkers, log_probs)
    if step % 100 == 0:
        walker_history.append(walkers.copy())
    if step % 100 == 0:
        print(f"Step {step}")

walker_history = np.array(walker_history)  #shape (10, 500, 6)

param_idx = 0  #plot evolution of zed

for i, step_walkers in enumerate(walker_history):
    plt.figure()
    plt.hist(step_walkers[:, param_idx], bins=50, alpha=0.7)
    plt.title(f"Distribution of param {param_idx} at step {i*100}")
    plt.xlabel(f"Parameter {param_idx} value")
    plt.ylabel("Count")
    plt.show()

plt.figure(figsize=(10, 6))
for i in range(nwalkers):  #plot random walkers for convergence of zed
    trace = [w[i, param_idx] for w in walker_history]
    plt.plot(np.arange(0, nsteps, 100), trace, alpha=0.5)
plt.xlabel("Iteration")
plt.ylabel(f"Parameter {param_idx} value")
plt.title("Trace plot for parameter 0")
plt.show()

flat_walkers = walker_history[-1].reshape(-1, ndim)
mean_params = np.mean(flat_walkers, axis=0)
print("Posterior mean parameters:", mean_params)

y_fit = fifth(mean_params, xpos)
sort_idx = np.argsort(xpos)
plt.scatter(xpos, ypos, s=10, label='data')
plt.plot(xpos[sort_idx], y_fit[sort_idx], 'r', label='MCMC mean fit')
plt.legend()
plt.title("Data and MCMC Mean Fit")
plt.show()

