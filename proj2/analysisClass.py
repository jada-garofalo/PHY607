import numpy as np
import matplotlib.pyplot as plt

class Analysis:
    """
    Analyze results of the simulation
    """
    def __init__(self, bodies):
        """
        Params
        ------
        bodies: list of Body objects from the simulation
        """
        self.bodies = bodies

    def total_energy(self):
        """
        Compute total system kinetic energy
        """
        return sum(body.compute_energy() for body in self.bodies)

    def total_momentum(self):
        """
        Compute total system momentum
        """
        total = np.zeros(3)
        for body in self.bodies:
            total = total +  body.compute_momentum()
        return total

    def plot_trajectories(self):
        """
        Plot trajectories of all bodies
        """
        for body in self.bodies:
            traj = body.get_trajectory_values()
            plt.plot(traj[:,0], traj[:,1])
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Trajectories of bodies")
        plt.show()

    def summarize(self):
        """
        Print or return statistics
        """
        energy = self.total_energy()
        momentum = self.total_momentum()
        print(f"Total kinetic energy: {energy:.3e}")
        print(f"Total momentum: {momentum}")

