import numpy as np
import matplotlib.pyplot as plt

def rlc_circuit(method='Euler', L, R, C, id1_init, total_t, dt, plot=True):
    '''
    Numerically solves for current as a function of time in an RLC circuit

    Params
    method: string, algorithm for numerical calculation (Euler (default), Symplectic, RK2)
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
    
    if method=='Euler':
        for n in range(n_steps - 1):
            id2[n] = -2 * alpha * id1[n] - omega_n**2 * id0[n]
            id1[n+1] = id1[n] + dt * id2[n]
            id0[n+1] = id0[n] + dt * id1[n]

    if plot:
        t_array = np.linspace(0, total_t, n_steps)
        plt.plot(t_array, id0)
        plt.plot
        plt.xlabel('Time (s)')
        plt.ylabel('Current i(t) (amps)')
        plt.title(f"RLC Circuit Current using {method} Algorithm")
        plt.grid(True)
        plt.show()

def linear_charge(method='Midpoint', lambda, length, dx, eval_points,plot=True):
    '''
    Numerically solves for the electric field due to a linear charge distribution
    
    Params
    method: string, algorithm for numerical integration (Left-Hand Riemann, Midpoint (default), Trapezoid)
    lamda: linear charge density, C/m
    length: length of charge distribution, m
    dx: step size
    eval_points: array of test points
    plot: boolean, if True (default) plots

    Returns
    None
    '''
    eps_n = 8.854 * (10**(-12)) #vacuum permittivity
    n_points = length / dx
    E_arr = []
    for x_test in eval_points:
        E_total = 0
        for i in range(n_points):
            lower = i * dx
            upper = lower + dx
            if method == 'Left-hand Riemann':
                E_total = E_total + (lambda * (x_test - lower) / abs(x_test - lower)**3) * dx
            
            elif method == 'Midpoint':
                x_eval = lower + 0.5 * dx
                E_total = E_total + (lambda * (x_test - x_eval) / abs(x_test - x_eval)**3) * dx

            elif method == 'Trapezoid':
                E_left_comp = lambda * (x_test - lower) / abs(x_test - lower)**3
                E_right_comp = lambda * (x_test - upper) / abs(x_test - upper)**3
                E_total = E_total + 0.5 * (E_left_comp + E_right_comp) * dx

        E_total = E_total / (4 * np.pi * eps_n)
        E_arr.append(E_total)

    if plot:
        plt.plot(eval_points, E_array)
        plt.xlabel("x (m)")
        plt.ylabel("Electric Field (N/C)")
        plt.title(f"Electric Field due to Linear Charge using {method} Integration")
        plt.grid(True)
        plt.show()

