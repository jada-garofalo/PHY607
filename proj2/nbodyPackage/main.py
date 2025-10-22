"""
this main script will have: 

system initialization

the time step loop, in which there will be calls to the integration and interaction methods from the system class

"""
import numpy as np
import time as timer
from .systemClass import System
from .bodyClass import Body
from .analysisClass import Analysis

n_simulations = int(input("Enter number of simulations to perform: "))
# time the code
start_time_full = timer.time()
interaction_type_list = np.array([])
mass_transfer_list = np.array([])
for k in range(n_simulations):

    # time the code
    start_time = timer.time()

    # choose system parameters
    n_bodies = 10
    total_time = 50000
    time_step = 0.1
    gravity_constant = 6.6743 * 10**(-11) 
    dimensions = 2 # 2 or 3 for 2D or 3D motion
    interaction_distance = 0.01

    # choose limits for body mass and initial conditions
    mass_lim = np.array([0.1, 1])*100
    position0_lim = np.array([-1.0, 1.0])*1
    velocity0_lim = np.array([-1.0, 1.0])*0.0001

    # create system with the above parameters
    n_body_system = System(n_bodies, total_time, time_step, gravity_constant, 
                           dimensions, interaction_distance)

    # create array of bodies with random values for 
    bodies = np.array([])
    absorbed_bodies = np.array([])
    for i in range(n_bodies):
        mass = np.random.uniform(mass_lim[0], mass_lim[1], 1)
        position = np.random.uniform(position0_lim[0], position0_lim[1],dimensions)
        velocity = np.random.uniform(velocity0_lim[0], velocity0_lim[1],dimensions)
        bodies = np.append(bodies, Body(mass, position, velocity))

    print("-")
    print("Simulation", k+1, "Initial state:")
    system_analysis_start = Analysis(bodies)
    system_analysis_start.summarize(n_body_system)

    # integrate and update values
    iterations = round(total_time / time_step)
    time_list = np.linspace(0, total_time, iterations + 1)
    energies = np.zeros((iterations + 1, 3))
    energies[0, 0] = Analysis(bodies).total_kinetic_energy()[0]
    energies[0, 1] = np.sum(n_body_system.potential_energies(bodies))
    energies[0, 2] = energies[0, 0] + energies[0, 1]

    k_energies = np.zeros((iterations,n_bodies))
    p_energies = np.zeros((iterations,n_bodies))

    for i in range(iterations):
        x_new, v_new = n_body_system.integrate(bodies)
        for j in range(n_body_system.n_bodies):
            bodies[j].update_state(x_new[j, :], v_new[j, :])
            k_energies[i, j] = bodies[j].compute_energy()[0]
            #p_energies[i+1, j] = 
        interaction_list = n_body_system.interaction_detection(bodies)
        energies[i+1, 0] = Analysis(bodies).total_kinetic_energy()[0]
        energies[i+1, 1] = np.sum(n_body_system.potential_energies(bodies))
        energies[i+1, 2] = energies[i+1, 0] + energies[i+1, 1]
        if np.any(interaction_list > -1) == True:
            # ^ if there are entries to interaction_list
            
            # interact
            bodies, absorbed_bodies_out, interaction_types, mass_fractions = (
            n_body_system.interactions(bodies, interaction_list, 
                                       time_list[i+1]))
            absorbed_bodies = np.append(absorbed_bodies, absorbed_bodies_out)
            interaction_type_list = np.append(interaction_type_list,
                                              interaction_types)
            mass_transfer_list = np.append(mass_transfer_list, mass_fractions)
            
    all_bodies = np.append(bodies, absorbed_bodies)
    # time the code
    end_time = timer.time()
    print("-")
    print("runtime:", end_time-start_time)
    print("-")
    print("Simulation", k+1, "Final state:")

    # analyze...
    system_analysis_end = Analysis(bodies)
    system_analysis_end.summarize(n_body_system)
    system_analysis_end.plot_trajectories()
    system_analysis_end.system_energy_plot(energies, time_list)
    system_analysis_end.bodies_energy_plot(k_energies, p_energies, time_list)

### plot probabilities here
end_time_full = timer.time()
print("-")
print("runtime:", end_time_full-start_time_full)
print("-")
print(interaction_type_list)
system_analysis_end.measured_probabilities(interaction_type_list)
print(mass_transfer_list)

'''
system_analysis_trajectories = Analysis(all_bodies)
system_analysis_trajectories.plot_trajectories()

system_analysis_end.system_energy_plot(energies, time_list)
system_analysis_end.bodies_energy_plot(k_energies, p_energies, time_list)
'''

