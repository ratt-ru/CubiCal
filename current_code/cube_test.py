from __future__ import division
from time import time,sleep
from math import *
from cyfull import cycompute_jh, cycompute_jhr, cycompute_jhjinv, \
                   cycompute_update, cycompute_residual, cycompute_jhj, \
                   reduce_obs
import numpy as np

def create_data(n_dir=1, n_tim=1, n_fre=1, n_ant=3, t_int=1, f_int=1,
                d_type=np.complex64):

    vis_dims = [n_dir, n_tim, n_fre, n_ant, n_ant, 2, 2]

    vis = np.empty(vis_dims, dtype=d_type)

    vis[...] = np.random.random(vis_dims) + 1j*np.random.random(vis_dims)

    r_ind, c_ind = np.tril_indices(n_ant,-1)
    dr_ind, dc_ind = np.diag_indices(n_ant)

    vis[:,:,:,dr_ind,dc_ind,:,:] = 0
    vis[:,:,:,r_ind,c_ind,:,:] = vis[:,:,:,c_ind,r_ind,:,:].transpose(
                                                           0,1,2,3,5,4).conj()

    gain_dims = [n_dir, ceil(n_tim/t_int), ceil(n_fre/f_int), n_ant, 2, 2]

    gains = np.empty(gain_dims, dtype=d_type)

    gains[...] = np.random.random(gain_dims) + 1j*np.random.random(gain_dims)
    
    obs_dims = [n_tim, n_fre, n_ant, n_ant, 2, 2]

    obs = np.empty(obs_dims, dtype=d_type)

    obs[...] = np.random.random(obs_dims) + 1j*np.random.random(obs_dims)

    r_ind, c_ind = np.tril_indices(n_ant,-1)
    dr_ind, dc_ind = np.diag_indices(n_ant)

    obs[:,:,dr_ind,dc_ind,:,:] = 0
    obs[:,:,r_ind,c_ind,:,:] = obs[:,:,c_ind,r_ind,:,:].transpose(
                                                           0,1,2,4,3).conj()

    return vis, gains, obs

def compute_jh(model, gains, t_int, f_int):

    n_dir, n_tim, n_fre, n_ant, n_ant, n_cor, n_cor = model.shape

    jh = np.zeros_like(model)

    for d in xrange(n_dir):
        for t in xrange(n_tim):
            rt = t//t_int
            for f in xrange(n_fre):
                rf = f//f_int
                for aa in xrange(n_ant):
                    for ab in xrange(n_ant):
                        jh[d,t,f,aa,ab,:,:] = gains[d,rt,rf,aa,:,:].dot(
                                              model[d,t,f,aa,ab,:,:])

    return jh

def compute_jhr(jh, r, t_int, f_int):

    n_dir, n_tim, n_fre, n_ant, n_ant, n_cor, n_cor = jh.shape

    jhr = np.zeros([n_dir, ceil(n_tim/t_int), ceil(n_fre/f_int), n_ant, 2, 2],
                   dtype=jh.dtype)

    for d in xrange(n_dir):
        for t in xrange(n_tim):
            rt = t//t_int
            for f in xrange(n_fre):
                rf = f//f_int
                for aa in xrange(n_ant):
                    for ab in xrange(n_ant):
                        jhr[d,rt,rf,aa,:,:] += r[t,f,aa,ab,:,:].dot(
                                              jh[d,t,f,ab,aa,:,:])

    return jhr

def compute_jhj(jh, t_int=1, f_int=1):

    n_dir, n_tim, n_fre, n_ant, n_ant, n_cor, n_cor = jh.shape

    jhj = np.zeros([n_dir, ceil(n_tim/t_int), ceil(n_fre/f_int), n_ant, 2, 2],
                   dtype=jh.dtype)

    for d in xrange(n_dir):
        for t in xrange(n_tim):
            rt = t//t_int
            for f in xrange(n_fre):
                rf = f//f_int
                for aa in xrange(n_ant):
                    for ab in xrange(n_ant):
                        jhj[d,rt,rf,ab,:,:] += jh[d,t,f,aa,ab,:,:].T.conj().dot(
                                               jh[d,t,f,aa,ab,:,:])

    return jhj

def compute_jhjinv(jhj):

    n_dir, n_tim, n_fre, n_ant, n_cor, n_cor = jhj.shape

    jhjinv = np.empty_like(jhj)

    for d in xrange(n_dir):
        for t in xrange(n_tim):
            for f in xrange(n_fre):
                for aa in xrange(n_ant):
                    jhjinv[d,t,f,aa,:] = np.linalg.pinv(jhj[d,t,f,aa,:])

    return jhjinv

def compute_jhjinvjhr(jhjinv, jhr):

    n_dir, n_tim, n_fre, n_ant, n_cor, n_cor = jhr.shape

    update = np.empty_like(jhjinv)

    for d in xrange(n_dir):
        for t in xrange(n_tim):
            for f in xrange(n_fre):
                for aa in xrange(n_ant):
                    update[d,t,f,aa,:] = jhr[d,t,f,aa,:].dot(jhjinv[d,t,f,aa,:])

    return update

def compute_residual(obs, model, gains, t_int=1, f_int=1):

    n_dir, n_tim, n_fre, n_ant, n_ant, n_cor, n_cor = model.shape

    residual_shape = [ceil(n_tim/t_int), ceil(n_fre/f_int), n_ant, n_ant, 2, 2]
    residual = np.zeros(residual_shape, dtype=obs.dtype)

    for d in xrange(n_dir):
        for t in xrange(n_tim):
            rt = t//t_int
            for f in xrange(n_fre):
                rf = f//f_int
                for aa in xrange(n_ant):
                    for ab in xrange(n_ant):
                        residual[rt,rf,aa,ab,:] += obs[t,f,aa,ab,:] - \
                                                   gains[d,rt,rf,aa,:].dot(
                                                   model[d,t,f,aa,ab,:]).dot(
                                                   gains[d,rt,rf,ab,:].T.conj())

    return residual

if __name__=="__main__":

    n_dir = 1
    n_tim = 9
    n_fre = 9
    n_ant = 3
    d_type = np.complex128

    t_int = 2
    f_int = 2

    vis, gains, obs = create_data(n_dir, n_tim, n_fre, n_ant, t_int,
                                  f_int, d_type)

    cyjh = np.zeros(vis.shape, dtype=d_type)

    cycompute_jh(vis, gains, cyjh, t_int, f_int)

    jh = compute_jh(vis, gains, t_int, f_int)

    jhr = compute_jhr(jh, obs, t_int, f_int)

    cyjhr = np.zeros([n_dir, ceil(n_tim/t_int), ceil(n_fre/f_int), n_ant, 2, 2],
                     dtype=d_type)

    cycompute_jhr(cyjh, obs, cyjhr, t_int, f_int)

    jhj = compute_jhj(jh, t_int, f_int)

    jhjinv = compute_jhjinv(jhj)

    cyjhj = np.zeros([n_dir,  ceil(n_tim/t_int), ceil(n_fre/f_int), n_ant, 2, 2],
                        dtype=d_type)

    cycompute_jhj(cyjh, cyjhj, t_int, f_int)

    cyjhjinv = np.empty([n_dir,  ceil(n_tim/t_int), ceil(n_fre/f_int), n_ant, 2, 2],
                        dtype=d_type)

    cycompute_jhjinv(cyjhj, cyjhjinv)

    update = compute_jhjinvjhr(jhjinv, jhr)

    cyupdate = np.empty_like(cyjhjinv)

    cycompute_update(cyjhr, cyjhjinv, cyupdate)

    residual = compute_residual(obs, vis, gains, t_int, f_int)

    cyresidual = np.zeros_like(residual)

    # reduce_obs(obs, cyresidual, t_int, f_int)

    cycompute_residual(vis, gains, gains.transpose(0,1,2,3,5,4).conj(),
                       obs, cyresidual, t_int, f_int)

    print np.allclose(cyjh,jh)
    print np.allclose(cyjhr,jhr)
    print np.allclose(cyjhj, jhj)
    print np.allclose(cyjhjinv, jhjinv)
    print np.allclose(cyupdate, update)
    print np.allclose(cyresidual, residual)

    # print residual
    # print cyresidual

    # print cyresidual
    # print residual


    # print cyjhjinv - jhjinv


