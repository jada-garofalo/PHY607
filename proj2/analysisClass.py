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

    def total_kinetic_energy(self):
        """
        Compute total system kinetic energy
        """
        return sum(body.compute_energy() for body in self.bodies)

    def total_momentum(self):
        """
        Compute total system momentum
        """
        total = np.zeros(len(self.bodies[0].position))
        for body in self.bodies:
            total += body.compute_momentum()
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

    def summarize(self,system):
        """
        Print or return statistics
        """
        k_energy = self.total_kinetic_energy()
        p_energy = sum(system.potential_energies(self.bodies))
        momentum = self.total_momentum()
        print(f"Total kinetic energy: {k_energy}")
        print(f"Total potential energy: {p_energy}")
        print(f"Total energy: {k_energy+p_energy}")
        print(f"Total momentum: {momentum}")
    
    def system_energy_plot(self,energies,time):
        """
        Plot energies vs time
        """
        plt.plot(time,energies[:,0],label="kinetic")
        plt.plot(time,energies[:,1],label="potential")
        plt.plot(time,energies[:,2],label="total")
        plt.xlabel("time")
        plt.ylabel("energy")
        plt.title("Energy of system")
        plt.legend()
        plt.show()
        
    def bodies_energy_plot(self,k_energies,p_energies,time):
        """
        Plot energies vs time
        """
        plt.plot(time[1:],p_energies,label="potential")
        plt.plot(time[1:],k_energies,label="kinetic")
        plt.xlabel("time")
        plt.ylabel("energy")
        plt.title("Kinetic and potential energies of bodies")
        plt.show()
