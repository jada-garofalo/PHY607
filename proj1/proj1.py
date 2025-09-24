import numpy as np
import matplotlib.pyplot as plt

def rlc_circuit(L, R, C, id1_init, total_t, dt, method='Euler', plot=True, returns=False):
    '''
    Numerically solves for current as a function of time in an RLC-series circuit

    Params
    L: inductance, henrys
    R: resistance, ohms
    C: capacitance, farads
    id1_init: initial change in current, amps per second
    total_t: total amount of time to sample over
    dt: step size, seconds
    method: string, algorithm for numerical calculation (Euler (default), Symplectic, RK2, RK4)
    plot: boolean, if True (default) generates plot of current and its derivatives as functions of time
    returns: boolean, if True (non-default) returns id0, id1, id2, t_array arrays for verification

    Returns
    None (default)
        
    '''
    alpha = R / (2 * L)
    omega_n = (L*C)**(-0.5)
    n_steps = int(total_t / dt)
    t_array = np.linspace(0, total_t, n_steps)
    
    id0 = np.zeros(n_steps) #0th derivative of current
    id1 = np.zeros(n_steps) #1st derivative of current
    id2 = np.zeros(n_steps) #2nd derivative of current

    id1[0] = id1_init
    
    if method=='Euler':
        for n in range(n_steps - 1):
            id2[n] = -2 * alpha * id1[n] - omega_n**2 * id0[n]
            id1[n+1] = id1[n] + dt * id2[n]
            id0[n+1] = id0[n] + dt * id1[n]

    elif method=='Symplectic':
        for n in range(n_steps - 1):
            id2[n] = -2 * alpha * id1[n] - omega_n**2 * id0[n]
            id1[n+1] = id1[n] + dt * id2[n]
            id0[n+1] = id0[n] + dt * id1[n+1]
 
    elif method=='RK2':
        for n in range(n_steps - 1):
            id0_mid = id0[n] + 0.5 * dt * id1[n]
            id1_mid = id1[n] + 0.5 * dt * id2[n]
            id2_mid = -2 * alpha * id1_mid - omega_n**2 * id0_mid

            id0[n+1] = id0[n] + dt * id1_mid
            id1[n+1] = id1[n] + dt * id2_mid

    elif method=='RK4':
        for n in range(n_steps - 1):
            k1_d1 = id1[n]
            k1_d2 = -2 * alpha * id1[n] - omega_n**2 * id0[n]

            k2_d1 = id1[n] + 0.5 * dt * k1_d2
            k2_d2 = -2 * alpha * (id1[n] + 0.5 * dt * k1_d2) - omega_n**2 * (id0[n] + 0.5 * dt * k1_d1)

            k3_d1 = id1[n] + 0.5 * dt * k2_d2
            k3_d2 = -2 * alpha * (id1[n] + 0.5 * dt * k2_d2) - omega_n**2 * (id0[n] + 0.5 * dt * k2_d1)

            k4_d1 = id1[n] + dt * k3_d2
            k4_d2 = -2 * alpha * (id1[n] + dt * k3_d2) - omega_n**2 * (id0[n] + dt * k3_d1)

            id0[n+1] = id0[n] + (dt/6.0) * (k1_d1 + 2*k2_d1 + 2*k3_d1 + k4_d1)
            id1[n+1] = id1[n] + (dt/6.0) * (k1_d2 + 2*k2_d2 + 2*k3_d2 + k4_d2)

    else:
        print("Please use a valid method input!")
    
    if returns:
        return id0, id1, id2, t_array

    if plot:
        plt.plot(t_array, id0)
        plt.plot
        plt.xlabel('Time (s)')
        plt.ylabel('Current i(t) (amps)')
        plt.title(f"RLC Circuit Current using {method} Algorithm")
        plt.grid(True)
        plt.show()

def linear_charge(lambda_lin, length, dx, eval_points, method='Midpoint', plot=True, returns=False):
    '''
    Numerically solves for the electric field due to a linear charge distribution
    
    Params
    lamda_lin: linear charge density, Coulombs per meter
    length: length of charge distribution, meters (note, charge distribution will lay between x=0 and x=length)
    dx: step size, meters
    eval_points: array of test points
        for example, eval_points = np.linspace(-1.0, 2.0, 200)
    method: string, algorithm for numerical integration (Left-Hand Riemann, Midpoint (default), Trapezoid, Simpson)
    plot: boolean, if True (default) plots
    returns: boolean, if True (non-default) returns E_arr array for verification

    Returns
    None (default)
    See returns param
    '''
    eps_n = 8.854 * (10**(-12)) #vacuum permittivity
    n_points = int(length / dx)
    E_arr = []
    for x_test in eval_points:
        E_total = 0
        for i in range(n_points):
            lower = i * dx
            upper = lower + dx
            if method == 'Left-hand Riemann':
                E_total = E_total + (lambda_lin * (x_test - lower) / abs(x_test - lower)**3) * dx
            
            elif method == 'Midpoint':
                x_eval = lower + 0.5 * dx
                E_total = E_total + (lambda_lin * (x_test - x_eval) / abs(x_test - x_eval)**3) * dx

            elif method == 'Trapezoid':
                E_left_comp = lambda_lin * (x_test - lower) / abs(x_test - lower)**3
                E_right_comp = lambda_lin * (x_test - upper) / abs(x_test - upper)**3
                E_total = E_total + 0.5 * (E_left_comp + E_right_comp) * dx

            elif method=='Simpson':
                E_left_trap_comp = lambda_lin * (x_test - lower) / abs(x_test - lower)**3
                E_right_trap_comp = lambda_lin * (x_test - upper) / abs(x_test - upper)**3
                E_total_trap_comp = 0.5 * (E_left_trap_comp + E_right_trap_comp) * dx

                x_eval = lower + 0.5 * dx
                E_midpoint_comp = (lambda_lin * (x_test - x_eval) / abs(x_test - x_eval)**3) * dx

                E_total = E_total + (E_total_trap_comp / 3) + (2 * E_midpoint_comp / 3)

            else:
                print("Please use a valid method input!")

        E_total = E_total / (4 * np.pi * eps_n)
        E_arr.append(E_total)

    if returns:
        return E_arr

    if plot:
        plt.plot(eval_points, E_arr)
        plt.xlabel("x (m)")
        plt.ylabel("Electric Field (N/C)")
        plt.title(f"Electric Field due to Linear Charge using {method} Integration")
        plt.grid(True)
        plt.show()

