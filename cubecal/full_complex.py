from ReadModelHandler import *
from time import time,sleep
import math
import cyfull_complex as cyfull
import argparse
import sys
import cPickle
import concurrent.futures as cf
from Tools import logger
log = logger.getLogger("full_complex")

verbose_iterations = False

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
                chi_tol=1e-6, chi_interval=5, t_int=1, f_int=1, label=""):
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
    # S = slice(166,176)
    # print obser_arr.flatten()[S]
    # print model_arr.flatten()[S]
    # print residual.flatten()[S]
    # print abs(residual).max(),abs(residual).argmax()
    chi = np.sum(np.square(np.abs(residual)), axis=(-1,-2,-3,-4))
    init_chi = chi.mean()
    if verbose_iterations:
        print>> log, "{} initial chi-sq is {}".format(label, init_chi)

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
            if verbose_iterations:
                print>> log, "{} iteration {} chi-sq is {}, max gain update {}".format(label, iters, chi.mean(), diff_g.max())

            if n_conv/n_sols > 0.99:
                print>>log, "{} iteration {}: Static residual in {:.2%} of " \
                             "visibilities. Chisq {} -> {}".format(label, iters, n_conv/n_sols, init_chi, chi.mean())
                return gains

    print>>log, "{} iteration {}: Quorum {:.2%}. Chisq {} -> {}".format(label, iters, n_quor/n_sols, init_chi, chi.mean())

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

    g_inv = np.empty_like(gains)

    cyfull.cycompute_jhjinv(gains, g_inv) #Function can invert G.

    gh_inv = g_inv.transpose(0,1,2,3,5,4).conj()

    corr_vis = np.empty_like(obser_arr)

    cyfull.cycompute_corrected(obser_arr, g_inv, gh_inv, corr_vis, t_int, f_int)

    return corr_vis

def solve_and_save(obser_arr, model_arr, min_delta_g=1e-6, maxiter=30,
                   chi_tol=1e-6, chi_interval=5, t_int=1, f_int=1, label=""):

    gains = solve_gains(obser_arr, model_arr, min_delta_g, maxiter,
                        chi_tol, chi_interval, t_int, f_int, label=label)

    corr_vis = apply_gains(obser_arr, gains, t_int, f_int)

    return gains, corr_vis


# if __name__ == "__main__":

def debug():
    main(debugging=True)

def main(debugging=False):

    # init logger
    logger.enableMemoryLogging(True)

    if debugging:
        print>> log, "initializing from cubecal.last"
        args = cPickle.load(open("cubecal.last"))
        logger.logToFile(args.log, append=False)
    else:
        parser = argparse.ArgumentParser(description='Basic full-polarisation '
                                                     'calibration script.')
        parser.add_argument('msname', help='Name and location of MS.')
        parser.add_argument('smname', help='Name and location of sky model.',
                            type=str)
        parser.add_argument('-tc','--tchunk', type=int, default=1,
                            help='Determines time chunk size.')
        parser.add_argument('-fc','--fchunk', type=int, default=1,
                            help='Determines frequency chunk size.')
        parser.add_argument('--single-chunk-id', type=str,
                            help='Process only the specified chunk, then stop. Useful for debugging.')
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
        parser.add_argument('--ddid_to', type=int, help='Selects range from'
                            '--ddid to a particular DATA_DESC_ID.')
        parser.add_argument('-ddes','--use_ddes', action="store_true",
                            help='Simulate and solve for directions in sky model')
        parser.add_argument('-sim','--simulate', action="store_true",
                            help='Simulate visibilities using Montblanc.')
        parser.add_argument('-delg','--min_delta_g', type=float, default=1e-6,
                            help='Stopping criteria for delta G - stop when '
                                 'solutions change less than this value.')
        parser.add_argument('-delchi','--min_delta_chi', type=float, default=1e-6,
                            help='Stopping criteria for delta chi - stop when '
                                 'the residual changes less than this value.')
        parser.add_argument('-chiint','--chi_interval', type=int, default=5,
                            help='Interval at which to check the chi squared '
                                 'value - expensive computation.')
        parser.add_argument('-nproc','--processes', type=int, default=1,
                            help='Number of processes to run.')
        parser.add_argument('-savco','--save_corrected', action="store_true",
                            help='Save corrected visibilities to MS.')
        parser.add_argument('-weigh','--apply_weights', action="store_true",
                            help='Use weighted least squares.')
        parser.add_argument('-l', '--log', type=str, default="log",
                            help='Write output to log file.')

        args = parser.parse_args()

        cPickle.dump(args, open("cubecal.last","w"))

        logger.logToFile(args.log, append=False)
        print>> log, "started: " + " ".join(sys.argv)


    if args.ddid is not None:
        if args.ddid_to is not None:
            ddid =  args.ddid, args.ddid_to+1
        else:
            ddid = args.ddid
    else:
        ddid = None

    ms = ReadModelHandler(args.msname, args.smname, fid=args.field, ddid=ddid,
                          precision=args.precision, ddes=args.use_ddes,
                          simulate=args.simulate, apply_weights=args.apply_weights)
    print>>log, "reading MS columns"
    ms.mass_fetch()
    print>>log, "defining chunks"
    ms.define_chunk(args.tchunk, args.fchunk, single_chunk_id=args.single_chunk_id)

    if args.applyflags:
        ms.apply_flags = True
    if args.bitmask != 0:
        ms.bitmask = args.bitmask

    target = solve_and_save if args.save_corrected else solve_gains

    opts = { "min_delta_g"  : args.min_delta_g,
             "maxiter"      : args.maxiter,
             "chi_tol"      : args.min_delta_chi,
             "chi_interval" : args.chi_interval,
             "t_int"        : args.tint,
             "f_int"        : args.fint }


    t0 = time()

    # debugging mode: run serially (also if nproc not set)
    if debugging or not args.processes:
        for obser, model, weight, chunk_label in ms:
            gains = target(obser, model, label = chunk_label, **opts)
            ms.add_to_gain_dict(gains, [ms._chunk_ddid, ms._chunk_tchunk, ms._first_f, ms._last_f],
                                args.tint, args.fint)

    # normal mode: use futures to run in parallel
    else:
        with cf.ProcessPoolExecutor(max_workers=args.processes) as executor:
            future_gains = { executor.submit(target, obser, model, label=chunk_label, **opts) :
                             [ms._chunk_ddid, ms._chunk_tchunk, ms._first_f, ms._last_f]
                             for obser, model, weight, chunk_label in ms }

            for future in cf.as_completed(future_gains):

                if target is solve_and_save:
                    gains, covis = future.result()
                    ms.arr_to_col(covis, future_gains[future])
                else:
                    gains = future.result()

                ms.add_to_gain_dict(gains, future_gains[future],
                                    args.tint, args.fint)

    print>>log, ModColor.Str("Time taken: {} seconds".format(time() - t0), col="green")

    ms.write_gain_dict()
    
    if target is solve_and_save:
        ms.save(ms.covis, "CORRECTED_DATA")

    # t0 = time()
    # for obs, mod in ms:
    #     # print "Time: ({},{}) Frequncy: ({},{})".format(ms._first_t, ms._last_t,
    #     #                                                ms._first_f, ms._last_f)
    #     gains = solve_gains(obs, mod, args.min_delta_g, args.maxiter,
    #                         args.min_delta_chi, args.chi_interval,
    #                         args.tint, args.fint)
    #     corr_vis = apply_gains(obs, gains, t_int=args.tint, f_int=args.fint)
    #
    #     # ms.add_to_gain_dict(gains, t_int, f_int)
    #
    #     # ms.array_to_vis(corr_vis, ms._first_t, ms._last_t, ms._first_f, ms._last_f)
    # print "Time taken: {} seconds".format(time() - t0)


    # ms.save(ms.covis, "CORRECTED_DATA")
