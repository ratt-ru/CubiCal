from ReadModelHandler import *
from time import time,sleep
import math
import cyfull
import argparse
import MBTiggerSim as mbt
import TiggerSourceProvider as tsp

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

    tmp_array1 = np.empty([2,2], dtype=obser_arr.dtype)
    tmp_array2 = np.zeros(out_shape, dtype=obser_arr.dtype)

    cyfull.compute_jhr(obser_arr, gains, model_arr,
                       tmp_array1, tmp_array2, t_int, f_int)

    if (f_int>1) or (t_int>1):

        reduced_shape = list(tmp_array2.shape)
        reduced_shape[0] = int(math.ceil(reduced_shape[0]/t_int))
        reduced_shape[1] = int(math.ceil(reduced_shape[1]/f_int))

        interval_array = np.zeros(reduced_shape, dtype=obser_arr.dtype)
        cyfull.interval_reduce(tmp_array2, interval_array, t_int, f_int)
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

    jhjinv = np.zeros(jhjinv_shape, dtype=model_arr.dtype)

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

    if (f_int>1) or (t_int>1):

        n_dir, n_tim, n_fre, n_ant, n_cor, n_cor = gains.shape

        reduced_shape = [n_dir, n_tim, n_fre, n_ant, n_ant, n_cor, n_cor]

        m = np.zeros(reduced_shape, dtype=obser_arr.dtype)
        cyfull.model_reduce(model_arr, m, t_int, f_int)

        data = np.zeros(reduced_shape[1:], dtype=obser_arr.dtype)
        cyfull.model_reduce(obser_arr[np.newaxis,...], data[np.newaxis,...],
                            t_int, f_int)

        #TODO: 24//01/2017 - Write a single kernel for this operation.

        test = np.zeros_like(data)
        print test.dtype
        t0 = time()
        cyfull.cycompute_residual(m, gains, gains.transpose(0,1,2,3,5,4).conj(),
                                  test)
        print time() - t0

        gm = np.empty(reduced_shape, dtype=obser_arr.dtype)
        gmgh = np.empty(reduced_shape, dtype=obser_arr.dtype)

        cyfull.compute_bbyA(gains, m, gm, 1, 1)
        cyfull.compute_Abyb(gm, gains.transpose(0,1,2,4,3).conj(), gmgh, 1, 1)

    else:
        gm = np.empty_like(obser_arr)
        gmgh = np.empty_like(obser_arr)
        data = obser_arr

        cyfull.compute_bbyA(gains, model_arr, gm, t_int, f_int)
        cyfull.compute_Abyb(gm, gains.transpose(0,1,2,4,3).conj(), gmgh,
                            t_int, f_int)

    residual = data - gmgh

    return residual


def full_pol_phase_only(obser_arr, model_arr, min_delta_g=1e-6, maxiter=30,
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

    n_dir, n_tim, n_fre, n_ant, n_ant, n_cor, n_cor = model_arr.shape

    n_tim = int(math.ceil(n_tim/t_int))
    n_fre = int(math.ceil(n_fre/f_int))

    gain_shape = [n_dir, n_tim, n_fre, n_ant, n_cor, n_cor]

    gains = np.empty(gain_shape, dtype=obser_arr.dtype)
    gains[:] = np.eye(2)

    old_gains = np.empty_like(gains)
    old_gains[:] = np.inf
    n_quor = 0
    n_sols = float(n_dir*n_tim*n_fre)
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

        # print iters, n_quor/n_sols, n_quor

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
                print iters, "Static residual in {:.2%} of " \
                             "visibilities.".format(n_conv/n_sols)
                return gains

    print "Iteration {}: Quorum reached: {:.2%} solutions " \
                           "acceptable.".format(iters, n_quor/n_sols)

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


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Basic full-polarisation '
                                                 'calibration script.')
    parser.add_argument('msname', help='Name and location of MS.')
    parser.add_argument('-tc','--tchunk', type=int, default=1,
                        help='Determines time chunk size.')
    parser.add_argument('-fc','--fchunk', type=int, default=1,
                        help='Determines frequency chunk size.')
    parser.add_argument('-ti','--tint', type=int, default=1,
                        help='Determines time solution intervals.')
    parser.add_argument('-fi','--fint', type=int, default=1,
                        help='Determines frequency solution intervals.')
    parser.add_argument('-af','--applyflags', action="store_true",
                        help='Apply FLAG column to data.')
    parser.add_argument('-bm','--bitmask', type=int, default=0,
                        help='Apply masked bitflags to data.')
    parser.add_argument('-maxit','--maxiter', type=int, default=50,
                        help='Maximum number of iterations.')
    parser.add_argument('-f', '--field', type=int,
                        help='Selects a particular FIELD_ID.')
    parser.add_argument('-d', '--ddid', type=int,
                        help='Selects a particular DATA_DESC_ID.')
    parser.add_argument('-p', '--precision', type=str, default='32',
                        help='Selects a particular data type.')
    parser.add_argument('--ddid-to', type=int,
                        help='Selects range from --ddid to a particular DATA_DESC_ID.')

    args = parser.parse_args()

    if args.ddid is not None:
        if args.ddid_to is not None:
            ddid =  args.ddid, args.ddid_to+1
        else:
            ddid = args.ddid
    else:
        ddid = None

    ms = ReadModelHandler(args.msname, fid=args.field, ddid=ddid,
                          precision=args.precision)
    ms.mass_fetch()
    ms.define_chunk(args.tchunk, args.fchunk)

    if args.applyflags:
        ms.apply_flags = True
    if args.bitmask != 0:
        ms.bitmask = args.bitmask

    t_int, f_int = args.tint, args.fint

    t0 = time()
    for obs, mod in ms:
        print obs.shape, mod.shape
        print "Time: ({},{}) Frequncy: ({},{})".format(ms._first_t, ms._last_t,
                                                       ms._first_f, ms._last_f)
        gains = full_pol_phase_only(obs, mod, t_int=t_int, f_int=f_int,
                                    maxiter=args.maxiter)
        corr_vis = apply_gains(obs, gains, t_int=t_int, f_int=f_int)
        ms.array_to_vis(corr_vis, ms._first_t, ms._last_t, ms._first_f, ms._last_f)
    print "Time taken: {} seconds".format(time() - t0)

    ms.save(ms.covis, "CORRECTED_DATA")
