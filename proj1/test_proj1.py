import numpy as np
import proj1

def test_rlc_circuit():
    """
    Underdamped RLC (R small, L and C moderate): oscillatory decay
    Overdamped RLC (R large): no oscillation, exponential decay

    Uses RK2 and Symplectic models as examples
    """
    print("Underdamped RLC-series circuit (expect oscillatory decay)")
    proj1.rlc_circuit(L=1.0, R=0.5, C=0.25, id1_init=1.0, total_t=20.0, dt=0.01, method='RK2', plot=True)
    
    print("Overdamped RLC-series circuit (expect exponential decay)")
    proj1.rlc_circuit(L=1.0, R=5.0, C=0.25, id1_init=1.0, total_t=20.0, dt=0.01, method='Symplectic', plot=True)

def test_rlc_error_comparison():
    '''
    Test algorithm error for RLC series circuit across varying dt

    Expect more complex algoriths (e.g. RK4) to have better performance than more basic algorithms (e.g. Euler)
    '''
    L = 1.0
    C = 1.0
    R = 0.0
    omega_n = 1/(L*C)**0.5
    id1_init = 1.0
    total_t = 10.0
    methods = ['Euler', 'Symplectic', 'RK2', 'RK4']
    dts = [0.1, 0.01, 0.001, 0.0001]
    print("\nError comparison (RLC undamped LC)")
    for m in methods:
        print(f"\nMethod: {m}")
        for dt in dts:
            id0, id1, id2, t_array = proj1.rlc_circuit(L, R, C, id1_init, total_t, dt, method=m, plot=False, returns=True)
            exact = (id1_init/omega_n) * np.sin(omega_n * t_array)
            rmse = (np.nanmean((id0 - exact)**2))**0.5
            print(f"  dt={dt:.5f}, RMSE={rmse:.7e}")

def test_linear_charge():
    """
    Electric field should point away from positive charges
    Field magnitude largest near the ends of the charge distribution

    Length and eval_points below are chosen to fully contain a centered charge length, and close surrounding space
    """
    eval_points = np.linspace(-1.0, 2.0, 200)
    lambda_lin = 10**(-6)
    length = 1.0
    dx = 0.01
    
    print("Electric field using Midpoint integration (expect field to peak near edges)")
    proj1.linear_charge(lambda_lin=lambda_lin, length=length, dx=dx, eval_points=eval_points, method='Midpoint', plot=True)
    
    print("Electric field using Trapezoid integration (expect field to peak near edges)")
    proj1.linear_charge(lambda_lin=lambda_lin, length=length, dx=dx, eval_points=eval_points, method='Trapezoid', plot=True)

def test_linear_charge_error_comparison():
    """
    Compare numerical methods to analytic field at points outside the line
    """
    lambda_lin = 10**(-6)
    length = 1.0
    eval_points = np.concatenate([np.linspace(-10.0, -3.1, 50),
                                  np.linspace(3.1, 10.0, 50)])
    eps_n = 8.854 * 10**(-12)

    E_exact = []
    for x0 in eval_points:
        if x0 < 0:
            val = (lambda_lin / (4 * np.pi * eps_n)) * (1.0 / (x0 - length) - 1.0 / x0)
        elif x0 > length:
            val = (lambda_lin / (4 * np.pi * eps_n)) * (1.0 / x0 - 1.0 / (x0 - length))
        else:
            val = np.nan
        E_exact.append(val)
    np.array(E_exact)

    methods = ['Left-hand Riemann', 'Midpoint', 'Trapezoid', 'Simpson']
    dxs = [0.1, 0.01, 0.001, 0.0001]
    print("\nLinear charge convergence test")
    for m in methods:
        print(f"\nMethod: {m}")
        for dx in dxs:
            E_num = np.array(proj1.linear_charge(lambda_lin=lambda_lin, length=length, dx=dx, eval_points=eval_points, method=m, plot=False, returns=True))
            rmse = (np.nanmean((E_num - E_exact)**2))**0.5
            print(f"  dx={dx:.5f}, RMSE={rmse:.7e}")

test_rlc_circuit()
test_linear_charge()
test_rlc_error_comparison()
test_linear_charge_error_comparison()
