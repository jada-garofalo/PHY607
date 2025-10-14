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
        
        body i (bi) is the focused body
        body j (bj) is one of the other bodies
        a_i = 1/m_i * sum( G*m_i*m_j * (r_j-r_i) / (r_j-r_i)^3 )
        """
        accelerations = np.zeros((self.n_bodies,self.dim))
        for i in range(len(bodies)):
            acc_temp = np.zeros((1,self.dim))
            bi = bodies[i]
            for j in range(len(bodies)):
                if i == j: # skip when i and j reference the same body
                    continue
                bj = bodies[j]
                acc_temp += (self.G*bi.mass*bj.mass * 
                            (bj.position-bi.position) / 
                            np.linalg.norm(bj.position-bi.position)**3)
            accelerations[i,:] = acc_temp / bi.mass
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
        v = np.zeros((len(bodies),self.dim))
        x = np.zeros((len(bodies),self.dim))
        for i in range(len(bodies)):
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
