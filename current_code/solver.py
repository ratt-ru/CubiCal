from msread import *
from time import time, sleep

ms = DataHandler("WESTERBORK_GAP.MS")
ms.fetch_all()
ms.define_chunk(1,1)
a, b = ms.vis_to_array(0,105,0,1)

def compute_jhr(obser_arr, model_arr, gains, ind):

    spec_eye = np.zeros([2,4])
    spec_eye[(0,1),(0,3)] = 1

    new_shape = list(model_arr.shape)
    new_shape[-3:] = [4,1]

    GI = np.einsum("...ij,...jk->...ik", gains, spec_eye)

    RG = np.einsum("...ij,...jk->...ik", obser_arr[...,ind,:,:,:], gains)

    RGMH = np.einsum("...ij,...kj->...ik", RG, model_arr[...,:,ind,:,:])

    GIRGMH = np.einsum("...ij,...jk->...ik", GI, RGMH.reshape(new_shape))

    JHR = -2 * np.sum(GIRGMH.imag, axis=-3)

    return JHR

def compute_jhjinv(model_arr):

    new_shape = list(model_arr.shape)
    new_shape[-2:] = [4]

    to_norm = np.array([[2,0,0,0],[1,0,0,1],[1,0,0,1],[0,0,0,2]])

    jhjinv = np.sum(abs(model_arr.reshape(new_shape))**2, axis=-2).dot(to_norm)

    jhjinv[jhjinv!=0] = 1./jhjinv[jhjinv!=0]

    new_shape[-2:] = [2,2]

    return jhjinv.reshape(new_shape)

def compute_update(model_arr, obser_arr, gains):

    jhjinv = compute_jhjinv(model_arr)

    update = np.zeros_like(gains)

    for i in xrange(model_arr.shape[-3]):

        jhr = compute_jhr(obser_arr, model_arr, gains, i)

        update[...,i,:,:] = np.einsum("...ij,...jk->...ik", jhjinv[...,i,:,:], jhr)


        # print update

def full_pol_phase_only(model_arr, obser_arr):

    gains = np.empty(model_arr.shape[-3:])
    gains[:] = np.eye(2)

    for i in range(3):
        compute_update(model_arr, obser_arr, gains)


full_pol_phase_only(b, a)
# print compute_jhr(a, b, "a")
# print compute_jhjinv(b)











# def compute_jhjinv3(model_arr):
#
#     tst = np.sum(abs(model_arr)**2, axis=-3)
#
#     tst[...,0,0] = np.sum(tst[...,0,:] + tst[...,:,0], axis=-1)
#     tst[...,1,1] = np.sum(tst[...,1,:] + tst[...,:,1], axis=-1)
#     tst[...,(0,1),(1,0)] = 0
#
#     tst[tst!=0] = 1./tst[tst!=0]
#
#     return tst

# print b[0,0,1,0,:].reshape(2,2).dot(np.eye(2))

# CODE FOR COMPUTING JHR WITH EXPLICIT, INCONVENIENT KRON PROD.

# MCG = np.einsum("...ij,...jk->...ik", model_arr[...,:,0,:,:], gains)
#
# res = np.empty([14,4,4], dtype=np.complex128)
#
# for i in xrange(MCG.shape[-3]):
#     res[i,:,:] = np.kron(MCG[0,0,i,:,:], np.eye(2))
#
# res = np.einsum("...ij,...jk->...ik", GI, res)
#
# res = np.einsum("...ij,...jk->...ik", res, obser_arr[...,0,:,:,:].reshape(
#     blah))
#
# print -2 * np.sum(res, axis=-3).imag
