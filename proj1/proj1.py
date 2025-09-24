import numpy as np
import matplotlib.pyplot as plt

def rlc_circuit(L, R, C, id1_init, total_t, dt, plot=True):
    '''
    Numerically solves for current as a function of time in an RLC circuit

    Params
    L: inductance, H
    R: resistance, ohms
    C: capacitance, farads
    id1_init: initial change in current, amps per second
    total_t: total amount of time to sample over
    dt: step size
    plot: boolean, if True (default) generates plot of current and its derivatives as functions of time

    Returns
    None
    '''
    alpha = R / (2 * L)
    omega_n = (L*C)**(-0.5)
    n_steps = int(total_t / dt)

    id0 = np.zeros(n_steps)
    id1 = np.zeros(n_steps)
    id2 = np.zeros(n_steps)

    id1[0] = id1_init

    for n in range(n_steps - 1):
        id2[n] = -2 * alpha * id1[n] - omega_n**2 * id0[n]
        id1[n+1] = id1[n] + dt * id2[n]
        id0[n+1] = id0[n] + dt * id1[n]

    if plot:
        t_array = np.linspace(0, total_t, n_steps)
        plt.plot(t_array, id0)
        plt.plot
        plt.xlabel('Time [s]')
        plt.ylabel('Current i(t)')
        plt.title('RLC Circuit Current using Explicit Euler')
        plt.grid(True)
        plt.show()


