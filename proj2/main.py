"""
this main script will have: 

system initialization

the time step loop, in which there will be calls to the integration and interaction methods from the system class

"""

from systemClass import systemClass



# test code
system = systemClass(10, 100, 0.1)
print(system.time_step)
