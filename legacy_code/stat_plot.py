import pylab as plt
import numpy as np

def residual_contour(data, model):
    """
    Plots a contour map for the normed residual of baseline pq as a function
    of phi_p and phi_q.
    """

    phi_p = np.arange(-np.pi, np.pi, np.pi/180)
    phi_q = np.arange(-np.pi, np.pi, np.pi/180)

    phi_p_grid, phi_q_grid = np.meshgrid(phi_p,phi_q)
    phi_p_grid, phi_q_grid = np.exp(-1j*phi_p_grid), np.exp(1j*phi_q_grid)

    norm_map = np.square(np.abs(data - phi_p_grid*model*phi_q_grid))
    norm_map = np.flipud(norm_map)

    # return norm_map


    # plt.imshow(norm_map, interpolation="None", extent=[-np.pi, np.pi, -np.pi, np.pi])
    # plt.colorbar()
    # plt.show()

    return norm_map

def net_contour(obs_mat, mod_mat):

    ant_a = 0
    ant_b = 1

    net = 0

    for i in [ant_a, ant_b]:
        for j in range(obs_mat.shape[-1]):
            if i==j:
                continue
            if (ant_a == j) & (ant_b == i):
                continue
            if (ant_a == i) & (ant_b == j):
                net += residual_contour(obs_mat[...,i,j], mod_mat[...,i,j])
            if j>i:
                print j
                tmp = residual_contour(obs_mat[...,i,j], mod_mat[...,i,j])
                net += np.sum(tmp, axis=0)
                print np.sum(tmp, axis=0)[np.newaxis,:]
                plt.imshow(np.sum(tmp, axis=0))
                plt.show()
            if j<i:
                tmp = residual_contour(obs_mat[...,i,j], mod_mat[...,i,j])
                net += np.sum(tmp, axis=1)[:,np.newaxis]
                plt.imshow(net)
                plt.show()

    plt.imshow(tmp, interpolation="None")
    plt.show()


def add_plot(phi_p, phi_q, colour="k"):
    """
    Adds a point to a contour plot.
    """

    plt.plot(phi_p, phi_q, c=colour, marker="x")

    return

def show():
    """
    Display current plot.
    """

    plt.show()

    return