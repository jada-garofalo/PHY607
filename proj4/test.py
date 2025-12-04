from math import sqrt

from pde import PDE, CartesianGrid, MemoryStorage, ScalarField, plot_kymograph

grid = CartesianGrid([[0, 20]], 128, periodic=False)  # generate grid

# create a (normalized) wave packet with a certain form as an initial condition
initial_state = ScalarField.from_expression(grid, "exp(I * 5 * x) * exp(-(x - 10)**2)")
initial_state /= sqrt(initial_state.to_scalar("norm_squared").integral.real)

eq = PDE({"ψ": f"I * laplace(ψ)"})  # define the pde

# solve the pde and store intermediate data
storage = MemoryStorage()
eq.solve(initial_state, t_range=2.5, dt=1e-5, tracker=[storage.tracker(0.02)])

# visualize the results as a space-time plot
plot_kymograph(storage, scalar="norm_squared")
