"""
this main script will have: 

system initialization

the time step loop, in which there will be calls to the integration and interaction methods from the system class

"""
import numpy as np
import time as timer
from systemClass import System
from bodyClass import Body
from analysisClass import Analysis

# time the code
start_time = timer.time()

# choose system parameters
n_bodies = 5
total_time = 20000
time_step = 0.1
gravity_constant = 6.6743 * 10**(-11) 
dimensions = 2 # 2 or 3 for 2D or 3D motion
interaction_distance = 0.001

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
    position = np.random.uniform(position0_lim[0], position0_lim[1], dimensions)
    velocity = np.random.uniform(velocity0_lim[0], velocity0_lim[1], dimensions)
    bodies = np.append(bodies, Body(mass, position, velocity))

# integrate and update values
iterations = round(total_time/time_step)
for i in range(iterations):
    x_new, v_new = n_body_system.integrate(bodies)
    for j in range(n_body_system.n_bodies):
        bodies[j].update_state(x_new[j,:], v_new[j,:])
    interaction_list = n_body_system.interaction_detection(bodies)
    if np.any(interaction_list>-1) == True:
        # ^ if there are entries to interaction_list
        
        # interact
        bodies, absorbed_bodies_out = n_body_system.interactions(bodies, interaction_list)
        absorbed_bodies = np.append(absorbed_bodies, absorbed_bodies_out)
all_bodies = np.append(bodies, absorbed_bodies)






# time the code
end_time = timer.time()
print("runtime:", end_time-start_time)

# analyze...
system_analysis = Analysis(all_bodies)
system_analysis.plot_trajectories()

