from time import time,sleep
import math
from cyfull import cycompute_jh, cycompute_jhr, cycompute_jhjinv, \
                   cycompute_update, cycompute_residual
import numpy as np

def create_data(n_dir=1, n_tim=1, n_fre=1, n_ant=3, d_type=np.complex64):

    vis_dims = [n_dir, n_tim, n_fre, n_ant, n_ant, 2, 2]

    vis = np.empty(vis_dims, dtype=d_type)

    vis[...] = np.random.random(vis_dims) + 1j*np.random.random(vis_dims)

    r_ind, c_ind = np.tril_indices(n_ant,-1)
    dr_ind, dc_ind = np.diag_indices(n_ant)

    vis[:,:,:,dr_ind,dc_ind,:,:] = 0
    vis[:,:,:,r_ind,c_ind,:,:] = vis[:,:,:,c_ind,r_ind,:,:].transpose(
                                                           0,1,2,3,5,4).conj()

    gain_dims = [n_dir, n_tim, n_fre, n_ant, 2, 2]

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

def compute_jh(model, gains):

    n_dir, n_tim, n_fre, n_ant, n_ant, n_cor, n_cor = model.shape

    jh = np.zeros_like(model)

    for d in xrange(n_dir):
        for t in xrange(n_tim):
            for f in xrange(n_fre):
                for aa in xrange(n_ant):
                    for ab in xrange(n_ant):
                        jh[d,t,f,aa,ab,:,:] = gains[d,t,f,aa,:,:].dot(
                                              model[d,t,f,aa,ab,:,:])

    return jh

def compute_jhr(jh, r):

    n_dir, n_tim, n_fre, n_ant, n_ant, n_cor, n_cor = jh.shape

    jhr = np.zeros([n_dir, n_tim, n_fre, n_ant, 2, 2], dtype=jh.dtype)

    for d in xrange(n_dir):
        for t in xrange(n_tim):
            for f in xrange(n_fre):
                for aa in xrange(n_ant):
                    for ab in xrange(n_ant):
                        jhr[d,t,f,aa,:,:] += r[t,f,aa,ab,:,:].dot(
                                            jh[d,t,f,ab,aa,:,:])

    return jhr

def compute_jhjinv(jh):

    n_dir, n_tim, n_fre, n_ant, n_ant, n_cor, n_cor = jh.shape

    jhj = np.zeros([n_dir, n_tim, n_fre, n_ant, 2, 2], dtype=jh.dtype)
    jhjinv = np.empty_like(jhj)

    for d in xrange(n_dir):
        for t in xrange(n_tim):
            for f in xrange(n_fre):
                for aa in xrange(n_ant):
                    for ab in xrange(n_ant):
                        jhj[d,t,f,ab,:,:] += jh[d,t,f,aa,ab,:,:].T.conj().dot(
                                             jh[d,t,f,aa,ab,:,:])

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

def compute_residual(obs, model, gains):

    n_dir, n_tim, n_fre, n_ant, n_ant, n_cor, n_cor = model.shape

    residual = np.zeros_like(obs)

    for d in xrange(n_dir):
        for t in xrange(n_tim):
            for f in xrange(n_fre):
                for aa in xrange(n_ant):
                    for ab in xrange(n_ant):
                        residual[t,f,aa,ab,:] += gains[d,t,f,aa,:].dot(
                                                 model[d,t,f,aa,ab,:]).dot(
                                                 gains[d,t,f,ab,:].T.conj())

    return obs - residual

if __name__=="__main__":

    n_dir = 1
    n_tim = 1
    n_fre = 1
    n_ant = 3
    d_type = np.complex128

    vis, gains, obs = create_data(n_dir, n_tim, n_fre, n_ant, d_type)

    cyjh = np.zeros(vis.shape, dtype=d_type)

    cycompute_jh(vis, gains, cyjh, 1, 1)

    jh = compute_jh(vis, gains)

    jhr = compute_jhr(jh, obs)

    cyjhr = np.zeros([n_dir, n_tim, n_fre, n_ant, 2, 2], dtype=d_type)

    cycompute_jhr(cyjh, obs, cyjhr, 1, 1)

    jhjinv = compute_jhjinv(jh)

    cyjhjinv = np.zeros([n_dir, n_tim, n_fre, n_ant, 2, 2], dtype=d_type)

    cycompute_jhjinv(cyjh, cyjhjinv)

    update = compute_jhjinvjhr(jhjinv, jhr)

    cyupdate = np.empty_like(cyjhjinv)

    cycompute_update(cyjhr, cyjhjinv, cyupdate)

    residual = compute_residual(obs, vis, gains)

    cyresidual = obs.copy()

    cycompute_residual(vis, gains, gains.transpose(0,1,2,3,5,4).conj(),
                       cyresidual, np.ones_like(obs, dtype=np.float32), 1, 1)

    print np.allclose(cyjh,jh)
    print np.allclose(cyjhr,jhr)
    print np.allclose(cyjhjinv, jhjinv)
    print np.allclose(cyupdate, update)
    print np.allclose(cyresidual, residual)

    # print cyjhjinv - jhjinv