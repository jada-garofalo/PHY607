import numpy as np

class Body:
    """
    A class representing a single body in the simulation.
    
    Params
    ------
    mass: mass of the body
    position: 3D position vector of the body
    velocity: 3D velocity vector of the body
    
    Attr
    ------
    trajectory: stores past positions for trajectory tracking
    """

    def __init__(self, mass, position, velocity):
        self.mass = mass
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.trajectory = [self.position.copy()]

    def compute_energy(self):
        """
        Compute the kinetic energy of the body
        """
        return 0.5 * self.mass * np.dot(self.velocity, self.velocity)

    def compute_momentum(self):
        """
        Compute the linear momentum of the body
        """
        return self.mass * self.velocity

    def get_trajectory_values(self):
        """
        Return the list of positions
        """
        return np.array(self.trajectory)

    def update_state(self, new_position, new_velocity):
        """
        Update the body's state after one step
        """
        self.position = np.array(new_position)
        self.velocity = np.array(new_velocity)
        self.trajectory.append(self.position.copy())

