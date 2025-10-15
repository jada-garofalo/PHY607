import numpy as np

class System:
    """
    this class contains the functions needed for the overall system
    """
    
    def __init__(self, n_bodies, total_time, time_step, gravity_constant, dimensions):
        # system properties
        self.n_bodies = n_bodies
        self.total_time = total_time
        self.time_step = time_step
        self.G = gravity_constant
        self.dim = dimensions
    
    def accelerations(self, bodies):
        """
        this method finds the acceleration of each body due to gravity
        forces from all other bodies
        a_i = G * sum( m_j * (r_j-r_i) / (r_j-r_i)^3 )
        """
        positions = [bodies[0].position]
        masses = np.array([bodies[0].mass])
        accelerations = np.zeros((self.n_bodies,self.dim))
        for i in range(self.n_bodies-1):
            positions = np.append(positions, [bodies[i+1].position],axis=0)
            masses = np.append(masses, [bodies[i+1].mass], axis=0)
        for i in range(self.n_bodies):
            distances = positions - positions[i]
            distances_cubed = np.array([np.sum(distances**2,1)**1.5]).T
            distances_cubed[i] = 1 # dont divide by zero
            accelerations[i] = self.G * np.sum(masses * distances / 
                                               distances_cubed)
        return accelerations
    
    def integrate(self, bodies):
        """
        this method should integrate the odes for all bodies across a timestep,
        setting the updated trajectories.
        
        probably use rk4:
        x_new = x + dt/6 * (k1x + 2*k2x + 2*k3x + k4x)
        v_new = v + dt/6 * (k1v + 2*k2v + 2*k3v + k4v)
        """
        a = self.accelerations(bodies)
        #print("a", a)
        v = np.zeros((self.n_bodies,self.dim))
        x = np.zeros((self.n_bodies,self.dim))
        for i in range(self.n_bodies):
            v[i,:] = bodies[i].velocity
            x[i,:] = bodies[i].position
        v_new = v + a*self.time_step
        #print("v", v_new)
        x_new = x + v*self.time_step
        
        return x_new, v_new
    
    
    def interactions(self):
        """
        this method should determine the interaction type for any and all
        bodies within interaction distance, and compute and set the updated
        trajectories and masses for those bodies
        """
        temp = 0
