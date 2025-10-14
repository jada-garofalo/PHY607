"""
this main script will have: 

system initialization

the time step loop, in which there will be calls to the integration and interaction methods from the system class

"""
import numpy as np
from systemClass import System
from bodyClass import Body

# choose number of bodies, total sim time, and time step
n_bodies = 10
total_time = 10
time_step = 0.1

# create system with the above parameters
n_body_system = System(n_bodies, total_time, time_step)

# create array of bodies
bodies = np.array([])
for i in range(n_bodies):
    mass = np.random.uniform(0.1, 1.0, 1)
    position = np.random.uniform(-1, 1, 3)
    velocity = np.random.uniform(-1, 1, 3)
    bodies = np.append(bodies, Body(mass, position, velocity))


