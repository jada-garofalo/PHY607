import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# Data: (N bodies, runtime in seconds)
data = np.array([
    [1, 1.9315438270568848],
    [2, 3.6743006706237793],
    [5, 8.560064315795898],
    [10, 16.654868841171265],
    [25, 34.405126094818115],
    [50, 70.2809591293335],
    [75, 85.68523669242859],
    [100, 96.86922359466553]
])

N_bodies = data[:, 0]
runtimes = data[:, 1]

# === Fit a power law: runtime = a * N^b ===
def power_law(N, a, b):
    return a * N**b

params, _ = curve_fit(power_law, N_bodies, runtimes)
a, b = params

# Generate smooth curve for fit
N_fit = np.linspace(1, 100, 200)
runtime_fit = power_law(N_fit, a, b)

# === Plot ===
plt.figure(figsize=(8,6))
plt.scatter(N_bodies, runtimes, color='dodgerblue', s=80, label='Simulation data')
plt.plot(N_fit, runtime_fit, color='red', linestyle='--', linewidth=2,
         label=f'Fit: $t = {a:.2f} N^{{{b:.2f}}}$')

plt.title('Scaling of Simulation Runtime with Number of Bodies')
plt.xlabel('Number of Bodies (N)')
plt.ylabel('Runtime (s)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()

# === Optional log-log plot (common for scaling) ===
plt.figure(figsize=(8,6))
plt.loglog(N_bodies, runtimes, 'o', color='dodgerblue', label='Simulation data')
plt.loglog(N_fit, runtime_fit, 'r--', label=f'Fit: $t \\propto N^{{{b:.2f}}}$')

plt.title('Runtime Scaling (log-log view)')
plt.xlabel('log(Number of Bodies)')
plt.ylabel('log(Runtime)')
plt.grid(True, which='both', linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()

print(f"Fitted scaling relation: runtime ≈ {a:.3f} * N^{b:.3f}")

