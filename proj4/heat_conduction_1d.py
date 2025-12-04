import numpy as np
import matplotlib.pyplot as plt

# setup/input
n_elements = 20
temp_a = 10
temp_b = 20
dx = 0.1
temp_0 = 15
n_iterations = 100
alpha = 100

# maximum timestep for stability
dt = 0.5*(dx**2)/alpha
print('dt =', dt)

# initialize temperature variable
temp = np.zeros((n_iterations,n_elements+2))
temp[:,0] = temp_a
temp[:,n_elements+1] = temp_b
temp[0,1:n_elements] = temp_0

# initialize position and index lists
index_list = np.linspace(1,n_elements,n_elements)
position_list = np.linspace(0,dx*(n_elements+1),n_elements+2)

# step forward by dt, until n_iterations is reached
for n in range(n_iterations-1):
    for i in index_list:
        i = int(i)
        temp[n+1,i] = temp[n,i] + alpha*dt/(dx**2)*( temp[n,i+1] - 2*temp[n,i] + temp[n,i-1] )
    
    # plot temperature distribution for every time step
    plt.plot(position_list,temp[n,:])

# plot final temperature distribution
plt.plot   (position_list,temp[n+1,:]) 
plt.xlabel('x')
plt.ylabel('temp')
plt.show()


