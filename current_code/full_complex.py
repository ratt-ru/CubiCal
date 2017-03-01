from ReadModelHandler import *
from time import time,sleep
import math
import cyfull
import argparse
import MBTiggerSim as mbt
import TiggerSourceProvider as tsp
import cPickle

def compute_js(obser_arr, model_arr, gains, t_int=1, f_int=1):
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

    n_dir, n_tim, n_fre, n_ant, n_cor, n_cor = gains.shape

    jh = np.zeros_like(model_arr)

    cyfull.cycompute_jh(model_arr, gains, jh, t_int, f_int)

    jhr_shape = [n_dir, n_tim, n_fre, n_ant, n_cor, n_cor]

    jhr = np.zeros(jhr_shape, dtype=obser_arr.dtype)

    if n_dir > 1:
        r = compute_residual(obser_arr, model_arr, gains, t_int, f_int)
    else:
        r = obser_arr

    cyfull.cycompute_jhr(jh, r, jhr, t_int, f_int)

    jhj = np.zeros(jhr_shape, dtype=obser_arr.dtype)

    cyfull.cycompute_jhj(jh, jhj, t_int, f_int)

    jhjinv = np.empty(jhr_shape, dtype=obser_arr.dtype)

    cyfull.cycompute_jhjinv(jhj, jhjinv)

    return jhr, jhjinv

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


    jhr, jhjinv = compute_js(obser_arr, model_arr, gains, t_int, f_int)

    update = np.empty_like(jhr)

    cyfull.cycompute_update(jhr, jhjinv, update)

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

    n_dir, n_tim, n_fre, n_ant, n_cor, n_cor = gains.shape

    residual_shape = [n_tim, n_fre, n_ant, n_ant, n_cor, n_cor]

    residual = np.zeros(residual_shape, dtype=obser_arr.dtype)
    gains_h = gains.transpose(0,1,2,3,5,4).conj()

    cyfull.cycompute_residual(model_arr, gains, gains_h, obser_arr, residual,
                              t_int, f_int)

    return residual


def solve_gains(obser_arr, model_arr, min_delta_g=1e-6, maxiter=30,
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

    n_tim = int(math.ceil(float(n_tim)/t_int))
    n_fre = int(math.ceil(float(n_fre)/f_int))

    gain_shape = [n_dir, n_tim, n_fre, n_ant, n_cor, n_cor]

    gains = np.empty(gain_shape, dtype=obser_arr.dtype)
    gains[:] = np.eye(n_cor)

    old_gains = np.empty_like(gains)
    old_gains[:] = np.inf
    n_quor = 0
    n_sols = float(n_dir*n_tim*n_fre)
    iters = 1

    residual = compute_residual(obser_arr, model_arr, gains, t_int, f_int)

    chi = np.sum(np.square(np.abs(residual)), axis=(-1,-2,-3,-4))

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
                print "Iteration {}: Static residual in {:.2%} of " \
                             "visibilities.".format(iters, n_conv/n_sols)
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

    #TODO: VERY BROKEN!!!

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
    parser.add_argument('smname', help='Name and location of sky model.',
                        type=str)
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
    parser.add_argument('-ddes','--use_ddes', action="store_true",
                        help='Simulate and solve for directions in sky model')
    parser.add_argument('-sim','--simulate', action="store_true",
                        help='Simulate visibilities using Montblanc.')

    args = parser.parse_args()

    if args.ddid is not None:
        if args.ddid_to is not None:
            ddid =  args.ddid, args.ddid_to+1
        else:
            ddid = args.ddid
    else:
        ddid = None

    ms = ReadModelHandler(args.msname, args.smname, fid=args.field, ddid=ddid,
                          precision=args.precision, ddes=args.use_ddes,
                          simulate=args.simulate)
    ms.mass_fetch()
    ms.define_chunk(args.tchunk, args.fchunk)

    if args.applyflags:
        ms.apply_flags = True
    if args.bitmask != 0:
        ms.bitmask = args.bitmask

    t_int, f_int = args.tint, args.fint

    t0 = time()
    for obs, mod in ms:
        print "Time: ({},{}) Frequncy: ({},{})".format(ms._first_t, ms._last_t,
                                                       ms._first_f, ms._last_f)
        gains = solve_gains(obs, mod, t_int=t_int, f_int=f_int,
                                    maxiter=args.maxiter)
        ms.add_to_gain_dict(gains, t_int, f_int)
        # corr_vis = apply_gains(obs, gains, t_int=t_int, f_int=f_int)
        # ms.array_to_vis(corr_vis, ms._first_t, ms._last_t, ms._first_f, ms._last_f)
    print "Time taken: {} seconds".format(time() - t0)

    ms.write_gain_dict()
    # ms.save(ms.covis, "CORRECTED_DATA")
