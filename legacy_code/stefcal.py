import numpy as np
from numpy.core.umath import rad2deg
from data_handler_stef import *
import pylab as plt
import time


def stefcal(obs_vis, mod_vis, n_ant):

    sols = np.ones([n_ant,1], dtype=np.complex128)

    improvement = np.inf
    accuracy = 1e-6

    i = 1

    while improvement > accuracy:

        if (i%2) == 0:
            sols = 0.5*(sols+get_stef_updates(obs_vis, mod_vis, n_ant, sols))
        else:
            sols = get_stef_updates(obs_vis, mod_vis, n_ant, sols)

        # sols = get_stef_updates(obs_vis, mod_vis, n_ant, sols)

        sols = sols/np.abs(sols)


        r = stef_residual(obs_vis, mod_vis, n_ant, sols)
        # plt.scatter(i,sols[0,0]*sols[1,0].conj())
        plt.scatter(i,np.linalg.norm(r))
        # print "GN NORM = ", np.linalg.norm(r)

        i += 1

        if i>30:
            # print sols.real
            return sols


def get_stef_updates(vis_mat, mod_mat, n_ant, gain_solutions):

    updates = np.empty([n_ant,1], dtype=np.complex128)

    gain_solutions = gain_solutions.reshape([n_ant])

    for i in range(updates.shape[0]):
        gains = gain_solutions

        z = gains*mod_mat[..., i]

        numer = np.sum(vis_mat[..., i, :]*z)

        # denom = z.ravel().dot(z.ravel().T.conj())
        # print denom
        denom = np.sum(np.abs(z)**2)

        updates[i] = numer/denom

    return updates

def stef_residual(obs_vis, mod_vis, n_ant, sols):

    G = np.empty([n_ant,1], dtype=np.complex128)
    G[:] = sols

    r = obs_vis - G*mod_vis*G.T.conj()

    return r
