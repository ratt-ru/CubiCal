from cyfullms import *
from time import time,sleep
import math
import cykernels
import cyfull

def compute_jhr(obser_arr, model_arr, gains, t_int=1, f_int=1):
    """
    This function computes the (J^H)R term of the GN/LM method for the
    full-polarisation, phase-only case.

    Args:
        obser_arr (np.array): Array containing the observed visibilities.
        model_arr (np.array): Array containing the model visibilities.
        gains (np.array): Array containing the current gain estimates.

    Returns:
        jhr (np.array): Array containing the result of computing (J^H)R.
    """

    out_shape = list(obser_arr.shape)
    out_shape[-3:] = [2,2]

    tmp_array1 = np.empty([2,2], dtype=np.complex128)
    tmp_array2 = np.zeros(out_shape, dtype=np.complex128)

    blah = np.empty_like(obser_arr, dtype=np.complex128)
    blah2 = np.empty_like(obser_arr, dtype=np.complex128)
    cyfull.compute_Abyb(obser_arr, gains, blah, t_int, f_int)
    cyfull.compute_AbyA(blah, model_arr, blah2)

    #print blah2[0,10,0,:,0,0]

    cyfull.compute_jhr(obser_arr, gains, model_arr,
                       tmp_array1, tmp_array2, t_int, f_int)

    if (f_int>1) or (t_int>1):

        reduced_shape = list(tmp_array2.shape)
        reduced_shape[0] = int(math.ceil(reduced_shape[0]/t_int))
        reduced_shape[1] = int(math.ceil(reduced_shape[1]/f_int))

        interval_array = np.zeros(reduced_shape, dtype=np.complex128)
        cykernels.interval_reduce(tmp_array2, interval_array, t_int, f_int)
        tmp_array2 = interval_array

    jhr = tmp_array2

    return jhr


def compute_jhjinv(model_arr, gains, t_int=1, f_int=1):
    """
    This function computes the ((J^H)J)^-1 term of the GN/LM method for the
    full-polarisation, phase-only case. Note that this depends only on the
    model visibilities.

    Args:
        model_arr (np.array): Array containing the model visibilities.

    Returns:
        jhjinv (np.array): Array containing the result of computing ((J^H)J)^-1.
    """

    jhjinv_shape = list(model_arr.shape)
    jhjinv_shape[0] = int(math.ceil(jhjinv_shape[0]/t_int))
    jhjinv_shape[1] = int(math.ceil(jhjinv_shape[1]/f_int))

    jhjinv = np.zeros(jhjinv_shape, dtype=np.complex128)

    tmp_out = np.empty_like(model_arr)
    tmp_out2 = np.empty_like(model_arr)

    gains_h = gains.transpose(0,1,2,4,3).conj()

    cyfull.compute_Abyb(model_arr, gains_h, tmp_out, t_int, f_int)

    cyfull.compute_Abyb(tmp_out, gains, tmp_out2, t_int, f_int)

    cyfull.compute_AbyA(tmp_out2, model_arr, tmp_out)   

    cyfull.reduce_6d(tmp_out, jhjinv, t_int, f_int)

    jhjinv = np.sum(jhjinv, axis=-3)

    cyfull.invert_jhj(jhjinv)

    return jhjinv


def compute_update(model_arr, obser_arr, gains, t_int=1, f_int=1):
    """
    This function computes the update step of the GN/LM method. This is
    equivalent to the complete (((J^H)J)^-1)(J^H)R.

    Args:
        obser_arr (np.array): Array containing the observed visibilities.
        model_arr (np.array): Array containing the model visibilities.
        gains (np.array): Array containing the current gain estimates.
        jhjinv (np.array): Array containing (J^H)J)^-1. (Invariant)

    Returns:
        update (np.array): Array containing the result of computing
            (((J^H)J)^-1)(J^H)R
    """

    jhjinv = compute_jhjinv(model_arr, gains, t_int, f_int)

    jhr = compute_jhr(obser_arr, model_arr, gains, t_int, f_int)

    update = np.empty_like(jhr)

    cyfull.compute_update(jhr, jhjinv, update)

    return update


def compute_residual(obser_arr, model_arr, gains, t_int=1, f_int=1):
    """
    This function computes the residual. This is the difference between the
    observed data, and the model data with the gains applied to it.

    Args:
        obser_arr (np.array): Array containing the observed visibilities.
        model_arr (np.array): Array containing the model visibilities.
        gains (np.array): Array containing the current gain estimates.

    Returns:
        residual (np.array): Array containing the result of computing D-GMG^H.
    """

    # TODO: The solution invterval code here must be fixed. 

    if (f_int>1) or (t_int>1):

        reduced_shape = list(model_arr.shape)
        reduced_shape[0] = int(math.ceil(reduced_shape[0]/t_int))
        reduced_shape[1] = int(math.ceil(reduced_shape[1]/f_int))

        gmgh = np.zeros(reduced_shape, dtype=np.complex128)
        cykernels.model_reduce(model_arr, gmgh, t_int, f_int)

        data = np.zeros(reduced_shape, dtype=np.complex128)
        cykernels.model_reduce(obser_arr, data, t_int, f_int)

        cykernels.apply_gains(gains, gains.conj(), gmgh, gmgh, 1, 1)

    else:
        gm = np.zeros_like(obser_arr)
        gmgh = np.zeros_like(obser_arr)
        data = obser_arr

        cyfull.compute_bbyA(gains, model_arr, gm, t_int, f_int)
        cyfull.compute_Abyb(gm, gains.transpose(0,1,2,4,3).conj(), gmgh,
                            t_int, f_int)

    residual = data - gmgh

    return residual


def full_pol_phase_only(model_arr, obser_arr, min_delta_g=1e-6, maxiter=30,
                        chi_tol=1e-6, chi_interval=5, t_int=1, f_int=1):
    """
    This function is the main body of the GN/LM method. It handles iterations
    and convergence tests.

    Args:
        obser_arr (np.array): Array containing the observed visibilities.
        model_arr (np.array): Array containing the model visibilities.
        min_delta_g (float): Gain improvement threshold.
        maxiter (int): Maximum number of iterations allowed.
        chi_tol (float): Chi-squared improvement threshold.
        chi_interval (int): Interval at which the chi-squared test is performed.

    Returns:
        gains (np.array): Array containing the final gain estimates.
    """


    gain_shape = list(model_arr.shape)
    gain_shape[-3:] = [2, 2]
    gain_shape[0] = int(math.ceil(gain_shape[0]/t_int))
    gain_shape[1] = int(math.ceil(gain_shape[1]/f_int))

    gains = np.zeros(gain_shape, dtype=np.complex128)
    gains[...,(0,1),(0,1)] = 1

    old_gains = np.empty_like(gains)
    old_gains[:] = np.inf
    n_quor = 0
    n_sols = float(gain_shape[0]*gain_shape[1])
    iters = 1

    residual = compute_residual(obser_arr, model_arr, gains, t_int, f_int)

    chi = np.sum(np.square(np.abs(residual)), axis=(-1,-2,-3,-4))

    # chi = np.linalg.norm(residual, axis=(-4,-3))
    # chi = np.linalg.norm(chi, axis=(-2,-1))

    min_quorum = 0.99

    # TODO: Add better messages.

    while n_quor/n_sols < min_quorum:

        if iters % 2 == 0:
            gains = 0.5*(gains + \
                compute_update(model_arr, obser_arr, gains, t_int, f_int))
        else:
            gains = compute_update(model_arr, obser_arr, gains, t_int, f_int)

        diff_g = np.sum(np.square(np.abs(old_gains - gains)), axis=(-1,-2,-3))
        norm_g = np.sum(np.square(np.abs(gains)), axis=(-1,-2,-3))
        n_quor = np.sum(diff_g/norm_g <= min_delta_g**2)
        n_quor += np.sum(norm_g==0)
        

        old_gains = gains.copy()

        print iters, n_quor/n_sols, n_quor
        #print model_arr[0,10,0,:,0,0]

        iters += 1
        
        if iters ==maxiter:
            break

        if iters > maxiter:
            print "Maxiter exceeded."
            return gains

        if (iters % chi_interval) == 0:
            old_chi = chi

            residual = compute_residual(obser_arr, model_arr, gains,
                                                  t_int, f_int)

            chi = np.sum(np.square(np.abs(residual)), axis=(-1,-2,-3,-4))

            n_conv = float(np.sum(((old_chi - chi) < chi_tol)))

            if n_conv/n_sols > 0.99:
                print iters, "Static residual in {:.2%} of visibilities.".format(
                    n_conv/n_sols)
                return gains

    print iters, "Quorum reached: {:.2%} solutions acceptable.".format(n_quor/n_sols)
    return gains


def apply_gains(obser_arr, gains, t_int=1, f_int=1):
    """
    Applies the inverse of the gain estimates to the observed data matrix.

    Args:
        obser_arr (np.array): Array of the observed visibilities.
        gains (np.array): Array of the gain estimates.

    Returns:
        inv_gdgh (np.array): Array containing (G^-1)D(G^-H).
    """

    inv_gains = gains.copy()

    cyfull.invert_jhj(inv_gains)

    tmp_out = np.empty_like(obser_arr)

    cyfull.compute_bbyA(inv_gains, obser_arr, tmp_out, t_int, f_int)

    inv_gains = inv_gains.transpose(0,1,2,4,3).conj()

    inv_gdgh = np.empty_like(obser_arr)

    cyfull.compute_Abyb(tmp_out, inv_gains, inv_gdgh, t_int, f_int)

    return inv_gdgh

def expand_index(indices, t_int=1, f_int=1, t_lim=np.inf, f_lim=np.inf):

    new_ind_a = []
    new_ind_b = []

    if t_lim%t_int == 0:
        t_lim = np.inf
    if f_lim%f_int == 0:
        f_lim = np.inf

    if (t_lim != np.inf) or (f_lim != np.inf):
        for i in indices:
            for j in xrange(int(t_int)):
                tmp_t_ind = i[0] + j
                if tmp_t_ind >= t_lim:
                    break

                for k in xrange(int(f_int)):
                    tmp_f_ind = i[1] + k
                    if tmp_f_ind >= f_lim:
                        break

                    new_ind_a.append(tmp_t_ind)
                    new_ind_b.append(tmp_f_ind)

    else:
        for i in indices:
            for j in xrange(int(t_int)):
                for k in xrange(int(f_int)):
                    new_ind_a.append(i[0] + j)
                    new_ind_b.append(i[1] + k)

    return new_ind_a, new_ind_b

ms = DataHandler("~/MEASUREMENT_SETS/D147.sel.MS")
#ms = DataHandler("~/MEASUREMENT_SETS/3C147-LO4-4M5S.MS/SUBMSS/D147-LO-NOIFS-NOPOL-4M5S.MS")
#ms = DataHandler("WESTERBORK_POL.MS")
ms.fetch_all()
ms.define_chunk(3254, 64)
ms.apply_flags = True

t_int, f_int = 1., 1.

t0 = time()
for b, a in ms:
    gains = full_pol_phase_only(a, b, t_int=t_int, f_int=f_int, maxiter=100)
    corr_vis = apply_gains(b, gains, t_int=t_int, f_int=f_int)
    ms.array_to_vis(corr_vis, ms._first_t, ms._last_t, ms._first_f, ms._last_f)
print time() - t0

ms.save(ms.covis, "CORRECTED_DATA")
