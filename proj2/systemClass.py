class systemClass:
    """
    this class contains the functions needed for the overall system
    """
    
    def __init__(self, n_bodies, total_time, time_step):
        # system properties
        self.n_bodies = n_bodies
        self.total_time = total_time
        self.time_step = time_step
    
    def accelerations(self):
    
        temp = 0
    
    def integrate(self):
        """
        this method should integrate the odes for all bodies across a timestep,
        setting the updated trajectories.
        
        probably use rk4:
        x_new = x + dt/6 * (k1x + 2*k2x + 2*k3x + k4x)
        u_new = u + dt/6 * (k1u + 2*k2u + 2*k3u + k4u)
        """
        temp = 0
    
    
    def interactions(self):
        """
        this method should determine the interaction type for any and all
        bodies within interaction distance, and compute and set the updated
        trajectories and masses for those bodies
        """
        temp = 0
