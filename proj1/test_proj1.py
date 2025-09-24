import numpy as np
import proj1

def test_rlc_circuit():
    """
    Underdamped RLC (R small, L and C moderate): oscillatory decay
    Overdamped RLC (R large): no oscillation, exponential decay
    """
    print("Underdamped RLC-series circuit (expect oscillatory decay)")
    proj1.rlc_circuit(L=1.0, R=0.5, C=0.25, id1_init=1.0, total_t=20.0, dt=0.01, method='RK2', plot=True)
    
    print("Overdamped RLC-series circuit (expect exponential decay)")
    proj1.rlc_circuit(L=1.0, R=5.0, C=0.25, id1_init=1.0, total_t=20.0, dt=0.01, method='Symplectic', plot=True)

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

test_rlc_circuit()
test_linear_charge()

