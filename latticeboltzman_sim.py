import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg', depending on your setup
import matplotlib.pyplot as plt

plotevery = 100

def dist(X, Y, cx, cy):
    return np.linalg.norm([X - cx, Y - cy])

def main():
    Nx = 400
    Ny = 100
    tau = 0.53
    Nt = 3000

    # Lattice speeds and weights for the 9 points
    NL = 9
    csx = np.array([0, 0, 1, 1, 1, 0, -1, -1, -1])
    csy = np.array([0, 1, 1, 0, -1, -1, -1, 0, 1])
    weights = np.array([4/9, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36])

    # Initial conditions
    F = np.ones((Ny, Nx, NL)) + 0.01 * np.random.randn(Ny, Nx, NL)
    F[:, :, 3] = 2.3

    # Creating the circle obstacle
    obstacle = np.full((Ny, Nx), False)

    for Y in range(Ny):
        for X in range(Nx):
            if dist(X, Y, Nx // 4, Ny // 2) < 13:
                obstacle[Y, X] = True

    # Enable interactive mode
    plt.ion()

    # Main loop
    for it in range(Nt):

        F[:,-1,[6,7,8]]= F[:,-2,[6,7,8]]
        F[:,0,[2,3,4]]= F[:,1, [2,3,4]] 

        for i, cx, cy in zip(range(NL), csx, csy):
            F[:, :, i] = np.roll(F[:, :, i], cx, axis=1)
            F[:, :, i] = np.roll(F[:, :, i], cy, axis=0)

        # Collisions
        boundary_f = F[obstacle, :]
        boundary_f = boundary_f[:, [0, 5, 6, 7, 8, 1, 2, 3, 4]]  # set velocity to opposite node

        # Fluid variables
        rho = np.sum(F, axis=2)
        ux = np.sum(F * csx, axis=2) / rho
        uy = np.sum(F * csy, axis=2) / rho

        F[obstacle, :] = boundary_f
        ux[obstacle] = 0
        uy[obstacle] = 0  # fluid velocities are all 0 within the obstacle

        # Calculating collisions
        F_eq = np.zeros(F.shape)
        for i, cx, cy, w in zip(range(NL), csx, csy, weights):
            F_eq[:, :, i] = rho * w * (
                1 + 3 * (cx * ux + cy * uy) + 9 * (cx * ux + cy * uy)**2 / 2 - 3 * (ux**2 + uy**2) / 2)
        
        F += -1/tau * (F - F_eq)

        if it % plotevery == 0:
            dfydx = ux[2:, 1:-1] - ux[0:-2,1:-1]
            dfxdy= uy[1:-1, 2:] - uy[1:-1,0:-2]
            curl = dfydx -dfxdy

            plt.clf()  # Clear the plot
            plt.imshow(curl, cmap="bwr")
            plt.draw()  # Explicitly draw the plot
            plt.pause(0.01)

    # Final plot
    plt.ioff()  # Turn off interactive mode
    plt.show()

if __name__ == "__main__":
    main()
