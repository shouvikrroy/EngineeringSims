import numpy as np
import matplotlib.pyplot as plot


# properties of rod

a = 110
length = 50 #mm
time = 4 #seconds
nodes = 50 #discretized parts

# Initializing

dx = length / nodes
dy = length / nodes
dt = min( dx**2 / (4*a), dy**2 / (4*a)) # stability function Courant-Friedrichs-Lewy
t_nodes = int(time/dt)

u = np.zeros((nodes,nodes)) + 20 # plate is initially at 20 degrees C

u[0, :] = 100
u[-1, :] = 100

fig, axis = plot.subplots()

pcm = axis.pcolormesh(u, cmap=plot.cm.jet, vmin=0, vmax=100)
plot.colorbar(pcm, ax=axis)

#simulating

counter = 0

while counter < time:
    
    w=u.copy()

    for i in range (1, nodes - 1):
        for j in range (1, nodes -1):
            
            dd_ux=(w[i-1,j] - 2*w[i,j] + w[i+1, j])/dx**2
            dd_uy=(w[i,j-1] - 2*w[i,j] + w[i, j+1])/dy**2
            
            u[i,j] = dt * a *  (dd_ux+ dd_uy)+ (w[i,j])

    counter += dt

    #updating the sim
    pcm.set_array(u)
    axis.set_title("Distribution at t: {:.3f} [s]".format(counter))
    plot.pause(0.1)

plot.show()
