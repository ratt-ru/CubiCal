import numpy as np
from numpy.core.umath import rad2deg
from data_handler_stef import *
import pylab as plt
import time
from stefcal import *


def get_updates(vis_mat, mod_mat, n_ant, phase_solutions, denom):

    updates = np.empty([n_ant,1])

    phase_solutions = phase_solutions.reshape([n_ant])

    for i in range(updates.shape[0]):
        gains = np.exp(-1j*phase_solutions)

        numer = (-gains[i].conj()*np.sum(vis_mat[..., i, :]
                                 *(gains*mod_mat[..., i]))).imag

        updates[i] = numer/denom[i]

    return updates


def residual(obs_vis, mod_vis, n_ant, sols):

    G = np.empty([n_ant,1], dtype=np.complex128)
    G[:] = np.exp(-1j*sols)

    r = obs_vis - G*mod_vis*G.T.conj()

    return r


def get_amp_updates(vis_mat, mod_mat, n_ant, amp_solutions):

    updates = np.empty([n_ant,1])

    amp_solutions = amp_solutions.reshape([n_ant])

    for i in range(updates.shape[0]):
        gains = amp_solutions

        z = gains*mod_mat[..., i]

        numer = (np.sum(vis_mat[..., i, :]*z)).real
        denom = np.sum(np.square(np.abs(z)))

        updates[i] = numer/denom

    return updates


def amp_residual(obs_vis, mod_vis, n_ant, sols):

    G = np.empty([n_ant,1], dtype=np.complex128)
    G[:] = sols.reshape(-1,1)

    r = obs_vis - G*mod_vis*G.T.conj()

    return r

def phocal(obs_vis, mod_vis, n_ant):

    sols = np.ones([n_ant,1], dtype=np.complex128)

    improvement = np.inf
    accuracy = 1e-6

    i = 1

    while improvement > accuracy:

        # sols += get_updates(obs_vis, mod_vis, n_ant, sols, denom)
        if i%2 == 0:
            sols = (sols+get_amp_updates(obs_vis, mod_vis, n_ant, sols))/2
        else:
            sols = get_amp_updates(obs_vis, mod_vis, n_ant, sols)
        # r = residual(obs_vis, mod_vis, n_ant, sols)
        r = amp_residual(obs_vis, mod_vis, n_ant, sols)
        # plt.scatter(i,np.exp(-1j*(sols[0,0]-sols[1,0])),c="r")
        # plt.scatter(i, np.linalg.norm(r), c="r")
        print "GN NORM = ", np.linalg.norm(r)

        i += 1

        if i>30:
            return sols


def apply_sols(vis_mat, phases):

    g_inv = np.exp(1j*phases).reshape(-1,1)

    cor_mat = g_inv*vis_mat*g_inv.T.conj()

    return cor_mat


def do_chunk(d_feed, m_feed, ant_a, ant_b, t_ind, t_chunk_size, n_ant, t_a,
             t_b, f_a, f_b):

    t_sel = range(t_a, t_b)
    n_tim = len(t_sel)
    t_slice = np.any(t_ind == t_sel, 1)

    t_chunk = t_ind[t_slice].reshape(-1) - np.min(t_sel)
    d_chunk = d_feed[t_slice]
    m_chunk = m_feed[t_slice]
    a_chunk = ant_a[t_slice]
    b_chunk = ant_b[t_slice]

    obs_vis = tst(d_chunk, a_chunk, b_chunk, t_chunk, n_ant, n_tim, f_b - f_a)
    mod_vis = tst(m_chunk, a_chunk, b_chunk, t_chunk, n_ant, n_tim, f_b - f_a)

    phi = phocal(obs_vis, mod_vis, n_ant).reshape((1, -1))

    cor_chunk = apply_sols(obs_vis, phi)

    cor_vis = tst2(cor_chunk, a_chunk, b_chunk, t_chunk, n_ant, n_tim, f_b - f_a)

    return cor_vis


def main():

    MS = open_ms("~/MeasurementSets/WESTERBORK_AMP.MS")

    feeds = [0, 3]

    d = get_data(MS, "DATA")
    m = get_data(MS, "MODEL_DATA")
    t = get_data(MS, "TIME")

    t_ind = times_to_ind(t).reshape(-1, 1)

    ant_a = get_data(MS, "ANTENNA1")
    ant_b = get_data(MS, "ANTENNA2")

    n_ant = np.max((ant_a, ant_b)) + 1

    t_chunk_size = 1
    f_chunk_size = 1

    n_t_chunk = (np.max(t_ind) + 1)//t_chunk_size

    if np.fmod(np.max(t_ind) + 1, t_chunk_size) != 0:
        n_t_chunk += 1

    n_f_chunk = d.shape[1]//f_chunk_size

    if np.fmod(d.shape[1], f_chunk_size) != 0:
        n_f_chunk += 1

    t_sel = range(0, np.max(t_ind) + 1, t_chunk_size)
    t_sel.append(np.max(t_ind) + 1)

    f_sel = range(0, d.shape[1], f_chunk_size)
    f_sel.append(d.shape[1])

    t0 = time.time()

    for feed in feeds:
        for j in range(n_f_chunk):
            d_feed = d[:,f_sel[j]:f_sel[j+1],feed]\
                .reshape((-1, f_sel[j+1]-f_sel[j]))
            m_feed = m[:,f_sel[j]:f_sel[j+1],feed]\
                .reshape((-1, f_sel[j+1]-f_sel[j]))

            for i in range(n_t_chunk):

                tmp_vis = do_chunk(d_feed, m_feed, ant_a, ant_b, t_ind,
                                   t_chunk_size, n_ant, t_sel[i], t_sel[i+1],
                                   f_sel[j], f_sel[j+1])

                if (i==0):
                    cor_vis = tmp_vis
                else:
                    cor_vis = np.vstack((cor_vis, tmp_vis))

            d[:, f_sel[j]:f_sel[j+1], feed:feed+1] = cor_vis

    save_data(d, MS, "CORRECTED_DATA")

    print time.time() - t0


def jacobian(mod_vis, gains, n_ant):

    n_bl = n_ant*(n_ant-1)/2

    gains = gains.reshape(-1,1)

    jacob = np.zeros([2*n_bl, n_ant], dtype=np.complex128)

    for i in range(n_ant):
        offset = recsum(n_ant, i)
        for j in range(i+1, n_ant):
            jacob[offset+j-i-1,i] = mod_vis[0,0,i,j]*gains[j,0].conj()
            jacob[offset+j-i-1,j] = mod_vis[0,0,i,j]*gains[i,0]

    jacob[n_bl:,:] = jacob[:n_bl,:].conj()

    return jacob


def recsum(a,b):

    sum = 0
    for i in range(a-b,a):
        sum += i
    return sum

if __name__=="__main__":
    main()
