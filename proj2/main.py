"""
this main script will have: 

system initialization

the time step loop, in which there will be calls to the integration and interaction methods from the system class

"""
import numpy as np
from systemClass import System
from bodyClass import Body
from analysisClass import Analysis

# choose system parameters
n_bodies = 20
total_time = 10000
time_step = 0.1
gravity_constant = 6.6743 * 10**(-11) 
dimensions = 2 # 2 or 3 for 2D or 3D motion

# choose limits for body mass and initial conditions
mass_lim = np.array([0.1, 1])*1000
position0_lim = np.array([-1.0, 1.0])*1
velocity0_lim = np.array([-1.0, 1.0])*0.0001

# create system with the above parameters
n_body_system = System(n_bodies, total_time, time_step, gravity_constant, dimensions)

# create array of bodies with random values for 
bodies = np.array([])
for i in range(n_bodies):
    mass = np.random.uniform(mass_lim[0], mass_lim[1], 1)
    position = np.random.uniform(position0_lim[0], position0_lim[1], dimensions)
    velocity = np.random.uniform(velocity0_lim[0], velocity0_lim[1], dimensions)
    bodies = np.append(bodies, Body(mass, position, velocity))

print("v0", bodies[0].velocity)

# integrate and update values
iterations = round(total_time/time_step)
for i in range(iterations):
    x_new, v_new = n_body_system.integrate(bodies)
    for j in range(len(bodies)):
        bodies[j].update_state(x_new[j,:], v_new[j,:])

print("vf", bodies[0].velocity)

# analyze...
system_analysis = Analysis(bodies)
system_analysis.plot_trajectories()

# debug section
#body1 = Body(1,[1,2,3],[4,5,6])
#print(body1.mass)
#print(body1.position)
#print(body1.velocity)
#print(body1.trajectory)
#print(body1.compute_energy())
#print(body1.get_trajectory_values())
#body1_analysis = Analysis([body1,body1])
#body1_analysis.plot_trajectories()
