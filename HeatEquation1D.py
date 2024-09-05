import numpy as np
import matplotlib.pyplot as plot


# properties of rod

a = 110
length = 50 #mm
time = 4 #seconds
nodes = 10 #discretized parts

# Initializing

dx = length / nodes
dt = 0.5 *(dx)**2 / a #Courant-Friedrichs-Lewy
t_nodes = int(time/dt)

u = np.zeros(nodes) + 20 # plate is initially at 20 degrees C

u[0] = 100
u[-1] = 100

fig, axis = plot.subplots()

pcm = axis.pcolormesh([u], cmap=plot.cm.jet, vmin=0, vmax=100)
plot.colorbar(pcm, ax=axis)
axis.set_ylim([-2,3]) #this will give the illusion of a 1d bar

#simulating

counter = 0

while counter < time:
    
    w=u.copy()

    for i in range (1, nodes - 1):
        u[i] = w[i] + (dt * a / dx**2) * (w[i-1] - 2 * w[i] + w[i+1])

    counter += dt

    #updating the sim
    pcm.set_array([u])
    axis.set_title("Distribution at t: {:.3f} [s]".format(counter))
    plot.pause(0.1)

plot.show()
