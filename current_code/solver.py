from msread import *
from time import time, sleep

ms = DataHandler("WESTERBORK_POINT.MS")
ms.fetch_all()
ms.define_chunk(1,1)
a, b = ms.vis_to_array(0,105,0,1)

def compute_jhr(obser_arr, model_arr, gains):

    gains = np.empty(model_arr.shape[-3:])
    gains[:] = np.eye(2)

    spec_eye = np.zeros([2,4])
    spec_eye[(0,1),(0,3)] = 1

    blah = list(model_arr.shape)
    blah[-3:] = [4,1]

    GI = np.einsum("...ij,...jk->...ik", gains, spec_eye)

    MH = np.transpose(model_arr[...,:,0,:,:], (0,1,2,4,3))

    RG = np.einsum("...ij,...jk->...ik", obser_arr[...,0,:,:,:], gains)

    RGMH = np.einsum("...ij,...jk->...ik", RG, MH)

    GIRGMH = np.einsum("...ij,...jk->...ik", GI, RGMH.reshape(blah))

    JHR = -2 * np.sum(GIRGMH, axis=-3).imag

    print JHR

    return JHR

compute_jhr(a, b, "a")
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
