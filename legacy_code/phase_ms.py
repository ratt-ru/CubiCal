import numpy as np
from numpy.core.umath import rad2deg
from data_handler import *
import pylab as plt
import time

def jacobian(obs_vis, mod_vis, n_ant, ant_aa, ant_bb, phase_errors):

    J = np.zeros((2*(np.shape(obs_vis)[0]), n_ant), dtype=np.complex128)

    for i in range(mod_vis.shape[0]):
        ant_a = ant_aa[i]
        ant_b = ant_bb[i]

        if ant_a==ant_b:
            continue

        row = recsum(n_ant - 1, ant_a) + ant_b - 1
        col = ant_a

        ant_a_dphi = np.exp(-1j*phase_errors[ant_a])
        ant_b_dphi = np.exp(-1j*phase_errors[ant_b])

        eq_conj = np.conj(ant_b_dphi)[0]
        ep_deri = -1j*ant_a_dphi[0]

        J[row, col] += mod_vis[i,0]*eq_conj*ep_deri

        row = recsum(n_ant - 1, ant_a) + ant_b - 1
        col = ant_b

        ep = ant_a_dphi[0]
        eq_conj_deri = np.conj(-1j*ant_b_dphi)[0]
        J[row, col] += mod_vis[i,0]*ep*eq_conj_deri

    J[(np.shape(obs_vis)[0]):,:] = np.conj(J[:(np.shape(obs_vis)[0]),:])

    return J

def recsum(a,b):

    sum = 0
    for i in range(a-b,a):
        sum += i
    return sum

def residuall(obs_vis, mod_vis, ant_a, ant_b, sols):

    r = obs_vis - np.exp(-1j*sols[ant_a])*mod_vis*np.exp(1j*sols[ant_b])

    r = np.vstack((r,r.conj()))

    return r

def gn_solver(obs_vis, mod_vis, n_ant, ant_a, ant_b):

    mod_vis = mod_vis[ant_a!=ant_b]
    obs_vis = obs_vis[ant_a!=ant_b]
    ant_a, ant_b = ant_a[ant_a!=ant_b], ant_b[ant_b!=ant_a]

    reorder = np.lexsort((ant_b[:],ant_a[:]))
    obs_vis = obs_vis[reorder]
    mod_vis = mod_vis[reorder]
    ant_a = ant_a[reorder]
    ant_b = ant_b[reorder]

    phase_solutions = np.zeros([n_ant,1], dtype=np.complex128)

    improvement = np.inf
    accuracy = 1e-6

    r = residuall(obs_vis, mod_vis, ant_a, ant_b, phase_solutions)

    i = 1

    while improvement > accuracy:
        J = jacobian(obs_vis, mod_vis, n_ant, ant_a, ant_b, phase_solutions)
        H = J.T.conj().dot(J)

        # plt.imshow(H.imag, interpolation="None")
        # plt.show()

        # H = np.diagflat(np.diagonal(H))

        # if (i%2)==0:
        #     phase_solutions += np.linalg.pinv(H).dot((J.T.conj()).dot(r))/2
        # else:
        phase_solutions += np.linalg.pinv(H).dot((J.T.conj()).dot(r))

        r = residuall(obs_vis, mod_vis, ant_a, ant_b, phase_solutions)

        print "GN NORM = ", np.linalg.norm(r)

        i += 1

        if i>10:
            print phase_solutions
            break

def get_updates(vis_mat, mod_mat, n_ant, phase_solutions):

    updates = np.empty([n_ant,1])

    phase_solutions = phase_solutions.reshape([n_ant])
    denom = np.sum(abs(mod_mat)**2, 1)

    for i in range(updates.shape[0]):
        gains = np.exp(-1j*phase_solutions)
        numer = (-gains[i].conj()*(vis_mat[i,:].dot(gains*mod_mat[:,i]))).imag

        updates[i] = numer/denom[i]

    return updates

def residual(obs_vis, mod_vis, n_ant, sols):

    G = np.zeros(obs_vis.shape, dtype=np.complex128)
    np.fill_diagonal(G, np.exp(-1j*sols))

    r = obs_vis - G.dot(mod_vis).dot(G.T.conj())

    return r

def phocal(obs_vis, mod_vis, n_ant):

    sols = np.zeros([n_ant,1], dtype=np.complex128)

    improvement = np.inf
    accuracy = 1e-6

    i = 1

    while improvement > accuracy:

        sols += get_updates(obs_vis, mod_vis, n_ant, sols)

        r = residual(obs_vis, mod_vis, n_ant, sols)

        print "GN NORM = ", np.linalg.norm(r)

        i += 1

        if i>100:
            print sols.real
            return sols

def inject_errors(vis, ant_a, ant_b, n_ant):

    errors = np.random.random([n_ant,1]).astype(np.complex128)

    print errors

    new_vis = np.empty_like(vis)

    for i in range(vis.shape[0]):
        ant_a_err = np.exp(-1j*errors[ant_a[i]])
        ant_b_err = np.exp( 1j*errors[ant_b[i]])

        new_vis[i] = ant_a_err*vis[i]*ant_b_err + 0.01*np.random.normal()

    return new_vis

if __name__=="__main__":

    MS = open_ms("~/MeasurementSets/WESTERBORK.MS")

    # D = get_data(MS, "DATA")[:,0,0].reshape((-1,1))
    D = get_data(MS, "DATA")
    D = get_stokes(D)

    # M = get_data(MS, "MODEL_DATA")[:,0,0].reshape((-1,1))
    M = get_data(MS, "MODEL_DATA")
    M = get_stokes(M)

    t_ind = times_to_ind(get_data(MS, "TIME"))

    ant_a = get_data(MS, "ANTENNA1")
    ant_b = get_data(MS, "ANTENNA2")

    n_ant = np.max((ant_a, ant_b)) + 1

    sel = 20

    D = D[t_ind==sel]
    M = M[t_ind==sel]
    ant_a = ant_a[t_ind==sel]
    ant_b = ant_b[t_ind==sel]

    D = inject_errors(M, ant_a, ant_b, n_ant)

    obs_vis = create_mat(D, ant_a, ant_b, n_ant)
    mod_vis = create_mat(M, ant_a, ant_b, n_ant)

    phocal(obs_vis, mod_vis, n_ant)
    gn_solver(D, M, n_ant, ant_a, ant_b)