from msread import *
from time import time, sleep

ms = DataHandler("WESTERBORK_GAP.MS")
ms.fetch_all()
ms.define_chunk(1,1)

def compute_jhr(obser_arr, model_arr, gains):

    spec_eye = np.zeros([2,4])
    spec_eye[(0,1),(0,3)] = 1

    new_shape = list(model_arr.shape)
    new_shape[-3:] = [4,1]

    RG = np.einsum("...ij,...jk->...ik", obser_arr, gains)

    RGMH = np.einsum("...ij,...kj->...ik", RG, model_arr.conj())

    RGMH = np.sum(RGMH, axis=-3)

    GHI = np.einsum("...ij,...jk->...ik", gains.conj(), spec_eye)

    GHIRGMH = np.einsum("...ij,...jk->...ik", GHI, RGMH.reshape(new_shape))

    JHR = -2 * GHIRGMH.imag

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

    jhr = compute_jhr(obser_arr, model_arr, gains)

    update = np.einsum("...ij,...jk->...ik", jhjinv, jhr)

    return update

def compute_residual(obser_arr, model_arr, gains):

    GM = np.einsum("...lmij,...lmjk->...mljk", gains, model_arr)
    GMGH = np.einsum("...lmki,...lmji->...lmkj", GM, gains.conj())

    return obser_arr - GMGH

def full_pol_phase_only(model_arr, obser_arr):

    phases = np.zeros([1, 1, 14, 2, 1])

    gains = np.einsum("...ij,...jk", np.exp(-1j*phases), np.ones([1,2]))
    gains[...,(0,1),(1,0)] = 0

    # print np.linalg.norm(compute_residual(obser_arr, model_arr, gains))

    for i in range(10):

        phases += compute_update(model_arr, obser_arr, gains)

        gains = np.einsum("...ij,...jk", np.exp(-1j*phases), np.ones([1,2]))
        gains[...,(0,1),(1,0)] = 0

        # print gains
        # print np.linalg.norm(compute_residual(obser_arr, model_arr, gains))

        # print gains

t0 = time()
for b,a in ms:
    full_pol_phase_only(a, b)
print time() - t0
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
