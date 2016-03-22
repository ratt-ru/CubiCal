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

    # MCG = np.einsum("...ij,...jk->...ik", model_arr[...,:,0,:,:], gains)

    GI = np.einsum("...ij,...jk->...ik", gains, spec_eye)

    MH = np.transpose(model_arr[...,:,0,:,:], (0,1,2,4,3))

    RG = np.einsum("...ij,...jk->...ik", obser_arr[...,0,:,:,:], gains)

    RGMH = np.einsum("...ij,...jk->...ik", RG, MH)

    blah = list(RGMH.shape)
    blah[-2:] = [4,1]

    GIRGMH = np.einsum("...ij,...jk->...i", GI, RGMH.reshape(blah))

    JHR = -2 * np.sum(GIRGMH, axis=-2).imag

    return JHR

    # print RGMH[:,:,:,(0,1),(0,1)]

    # print gains[1,:,:].dot(spec_eye).dot(np.kron(tst[1,:,:], np.eye(2)))

compute_jhr(a, b, "a")
# print b[0,0,1,0,:].reshape(2,2).dot(np.eye(2))