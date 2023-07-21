import numpy as np
import matplotlib.pyplot as plt


def solve_lorenz(N=10, max_time=4.0, sigma=10.0, beta=8.0/3, rho=28.0):
    """Plot solutions of the Lorenz attractor"""
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1], projection='3d')
    ax.axis('off')

    # Axis limits
    ax.set_xlim((-25, 25))
    ax.set_ylim((-35, 35))
    ax.set_zlim((5,55))

    def lorenz_deriv(x_y_z, t0, sigma=sigma, beta=beta, rho=rho):
        """ Compute the time evolution of a point in the Lorenz attractor"""
        x, y, z = x_y_z
        return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

    # Random starting point. The point is selected from a uniform distribution from -15 to 15
    np.random.seed(1)
    x0 = -15 + 30 * np.random.random((N, 3))