import numpy as np

class System:
    """
    this class contains the functions needed for the overall system
    """
    
    def __init__(self, n_bodies, total_time, time_step, gravity_constant, 
                 dimensions, interaction_distance):
        # system properties
        self.n_bodies = n_bodies
        self.total_time = total_time
        self.time_step = time_step
        self.G = gravity_constant
        self.dim = dimensions
        self.interaction_distance = interaction_distance
    
    def potential_energies(self, bodies):
        """
        this method returns a list of potential energy of each body
        """
        PE = np.zeros((self.n_bodies, 1))
        positions, masses = self.position_mass_list(bodies)
        for i in range(self.n_bodies):
            distances = positions - positions[i]
            distance_magnitudes = np.array([np.sum(distances**2,1)**0.5]).T
            interaction_distance_list = distance_magnitudes*0+self.interaction_distance
            distance_magnitudes[i] = np.inf # dont divide by zero
            interaction_distance_list[i] = np.inf
            PE[i] = self.G * masses[i] * np.sum(masses*(1/interaction_distance_list-1/distance_magnitudes)) 
        return PE
    
    def accelerations_solver(self, bodies):
        """
        this method finds the acceleration of each body due to gravity
        forces from all other bodies
        a_i = G * sum( m_j * (r_j-r_i) / (r_j-r_i)^3 )
        """
        
        accelerations = np.zeros((self.n_bodies, self.dim))
        positions, masses = self.position_mass_list(bodies)
        for i in range(self.n_bodies):
            distances = positions - positions[i]
            distances_cubed = np.array([np.sum(distances**2,1)**1.5]).T
            distances_cubed[i] = np.inf # dont divide by zero
            accelerations[i] = self.G * np.sum(masses * distances / 
                                               distances_cubed)
        return accelerations
    
    def position_mass_list(self, bodies):
        # returns lists of positions and masses in a math-ready format
        positions = np.array([bodies[0].position])
        masses = np.array([bodies[0].mass])
        for i in range(self.n_bodies-1):
            positions = np.append(positions, [bodies[i+1].position], axis=0)
            masses = np.append(masses, [bodies[i+1].mass], axis=0)
        return positions, masses
    
    def integrate(self, bodies):
        """
        this method numerically integrates the odes for all bodies across a 
        time step, giving the updated trajectories. (Euler's Method)
        """
        
        a = self.accelerations_solver(bodies)
        v = np.zeros((self.n_bodies, self.dim))
        x = np.zeros((self.n_bodies, self.dim))
        for i in range(self.n_bodies):
            v[i,:] = bodies[i].velocity
            x[i,:] = bodies[i].position
        v_new = v + a*self.time_step
        x_new = x + v_new*self.time_step
        
        return x_new, v_new
    
    def interaction_detection(self, bodies):
        """
        this method checks for interaction between bodies, and is intended to
        be used every iteration of the time loop.
        
        the output is a matrix containing which bodies are interacting, and is
        split up into rows for each interaction. The numbers in the matrix are
        the body numbers, where -1 means no body.
        """
        interaction_list = np.zeros((round(self.n_bodies/2),self.n_bodies),dtype=int)-1
        positions, _ = self.position_mass_list(bodies)
        for i in range(self.n_bodies):
            distance_vectors = positions - positions[i]
            distance_magnitudes = np.array([np.sum(distance_vectors**2,axis=1)**0.5]).T
            
            # temp list of interacting bodies
            temp_interact_list = np.where(distance_magnitudes < 
                                          self.interaction_distance)[0]
                                          
            # check if there are interactions
            if len(temp_interact_list) > 1:
                # ^ interaction threshold was met (more than one body listed)
                
                # check if interacting bodies are listed in interaction_list
                if np.any(interaction_list > -1) == False:
                    # ^ no entries in list, add a row with the interacting bodies
                    interaction_list[0,0:len(temp_interact_list)] = temp_interact_list
                    
                else:
                    # ^ there are entries in interaction_list already
                    
                    # check if any bodies arent already in interaction_list
                    mask1 = np.isin(temp_interact_list, interaction_list)
                    # check if the listed bodies are all in the same row
                    mask2 = np.isin(interaction_list, temp_interact_list)
                    temp_list1 = np.sum(mask2, axis=1)
                    if np.any(mask1==False) == False:
                        # ^ all current bodies are already in the list
                        
                        if np.sum(temp_list1 > 0) != 1:
                            # ^ they are not all in the same row
                            
                            # combine rows
                            temp_list2 = np.any(mask2, axis=1)
                            rows_to_combine = np.where(temp_list2)[0]
                            # make a temp array with the entries from each row that needs to be combined
                            temp_list3 = np.array([])
                            for j in range(len(rows_to_combine)):
                                mask = interaction_list[rows_to_combine[j],:] > -1
                                temp_list3 = np.append(temp_list3, interaction_list[rows_to_combine[j],:][mask])
                            
                            # set the first row from rows_to_combine to the contents of temp_list3
                            interaction_list[rows_to_combine[0],0:len(temp_list3)] = temp_list3
                            # then clear the other rows that are from temp_list3
                            interaction_list[rows_to_combine[1:],:] = -1
                        
                    else:
                        # ^ not all current bodies are in the list
                        
                        if np.sum(temp_list1 > 0) == 1:
                            # ^ they are all in the same row
                            
                            # add missing entries to that row:
                            # get an array of entries that need to be added to the row
                            entries_to_append = temp_interact_list[~mask1]
                            
                            # find which row is getting added to
                            row_to_append = np.where(mask2 == True)[0][0]
                            
                            # get the index of the first -1 in the row
                            append_index = np.where(interaction_list[row_to_append,:]==-1)[0][0]
                            
                            # insert the entries at that index
                            interaction_list[row_to_append,append_index:len(entries_to_append)] = entries_to_append
                            
                        else:
                            # ^ they are not all in the same row
                            
                            # combine rows, then add missing entries to that row:
                            # combine rows
                            temp_list2 = np.any(mask2, axis=1)
                            rows_to_combine = np.where(temp_list2)[0]
                            # make a temp array with the entries from each row that needs to be combined
                            temp_list3 = np.array([])
                            for j in range(len(rows_to_combine)):
                                mask = interaction_list[rows_to_combine[j],:] > -1
                                temp_list3 = np.append(temp_list3, interaction_list[rows_to_combine[j],:][mask])
                            # set the first row from rows_to_combine to the contents of temp_list3
                            interaction_list[rows_to_combine[0],0:len(temp_list3)] = temp_list3
                            # then clear the other rows that are from temp_list3
                            interaction_list[rows_to_combine[1:],:] = -1
                            
                            # then add missing entries to that row:
                            # rewrite mask2 with updated interaction_list
                            mask2 = np.isin(interaction_list, temp_interact_list)
                            # get an array of entries that need to be added to the row
                            entries_to_append = temp_interact_list[~mask1]
                            
                            # find which row is getting added to
                            row_to_append = np.where(mask2 == True)[0][0]
                            
                            # get the index of the first -1 in the row
                            append_index = np.where(interaction_list[row_to_append,:]==-1)[0][0]
                            
                            # insert the entries at that index
                            interaction_list[row_to_append,append_index:len(entries_to_append)] = entries_to_append        
        return interaction_list
    
    def interactions(self, bodies, interaction_list):
        """
        this method should determine the interaction type for any and all
        bodies within interaction distance, and compute and set the updated
        trajectories and masses for those bodies
        """
        
        absorbed_bodies = np.array([])
        n_interactions = np.sum(np.any(interaction_list>-1,axis=1))
        
        for i in range(n_interactions):
        
            body_index_list = interaction_list[i,interaction_list[0,:]>-1]
            print("interaction table:")
            print(interaction_list)
            # which bodies are interacting
            interacting_bodies = bodies[body_index_list]
            
            # number of interacting bodies
            n_interacting_bodies = len(interacting_bodies)
            
            # inverse cdf to choose which interaction method is used
            u = np.random.uniform()
            a = 2.5
            x = u**(1/a) # exponential dist
            if x <= 1/3:
                interaction_type = 3
                print("elastic collision interaction occured")
            elif x > 2/3:
                interaction_type = 1
                print("plastic collision interaction occured")
            else:
                interaction_type = 2
                print("elastic collision with partial mass transfer interaction occured")
            
            if interaction_type == 1:
                # fully plastic interaction
                # m1*v1 + m2*v2 = (m1+m2)*v3
                
                mass = 0
                momentum = 0
                mean_position = bodies[body_index_list[0]].position * 0
                for j in range(n_interacting_bodies):
                    mass += interacting_bodies[j].mass
                    momentum += (interacting_bodies[j].mass *
                                 interacting_bodies[j].velocity)
                    mean_position += interacting_bodies[j].position
                mean_position /= n_interacting_bodies
                velocity = momentum/mass
                # write over first interacting body
                bodies[body_index_list[0]].mass = mass
                bodies[body_index_list[0]].velocity = velocity
                bodies[body_index_list[0]].position = mean_position
                absorbed_bodies = np.append(absorbed_bodies, bodies[body_index_list[1:]])
                bodies = np.delete(bodies,body_index_list[1:])
                self.n_bodies -= n_interacting_bodies-1
                
            elif interaction_type == 2 and n_interacting_bodies == 2:
                # partial mass transfer fully elastic interaction
                # must be only 2 bodies interacting
                
                x1 = bodies[body_index_list[0]].position
                x2 = bodies[body_index_list[1]].position
                v1 = bodies[body_index_list[0]].velocity
                v2 = bodies[body_index_list[1]].velocity
                m1a = bodies[body_index_list[0]].mass
                m2a = bodies[body_index_list[1]].mass
                
                # pick mass transfer amount m3 by rejection sampling
                u = np.random.uniform()
                f = np.sin(np.pi*u)
                y = np.random.uniform(size=1000)
                p = sum(y<f)/len(y)
                print(p*100, "% mass transfer")
                m3 = p*m1a
                
                # masses after transfer
                m1b = m1a-m3
                m2b = m2a+m3
                
                # collision unit normal
                n = (x1-x2) / np.linalg.norm(x1-x2)
                # relative velocity
                vr = v1-v2
                
                # velocities after collision
                v1f = v1 - 2*m2b / (m1b+m2b) * np.dot(vr, n) * n
                v2f = v2 + 2*m1b / (m1b+m2b) * np.dot(vr, n) * n
                
                # update body properties
                bodies[body_index_list[0]].mass = m1b
                bodies[body_index_list[1]].mass - m2b
                bodies[body_index_list[0]].velocity = v1f
                bodies[body_index_list[1]].velocity = v2f
                
            elif interaction_type == 3 and n_interacting_bodies == 2:
                # fully elastic interaction
                # must be only 2 bodies interacting
                
                x1 = bodies[body_index_list[0]].position
                x2 = bodies[body_index_list[1]].position
                v1 = bodies[body_index_list[0]].velocity
                v2 = bodies[body_index_list[1]].velocity
                m1 = bodies[body_index_list[0]].mass
                m2 = bodies[body_index_list[1]].mass
                
                # collision unit normal
                n = (x1-x2) / np.linalg.norm(x1-x2)
                # relative velocity
                vr = v1-v2
                
                # velocities after collision
                v1f = v1 - 2*m2 / (m1+m2) * np.dot(vr, n) * n
                v2f = v2 + 2*m1 / (m1+m2) * np.dot(vr, n) * n
                
                # update body velocities
                bodies[body_index_list[0]].velocity = v1f
                bodies[body_index_list[1]].velocity = v2f
                
        return bodies, absorbed_bodies
