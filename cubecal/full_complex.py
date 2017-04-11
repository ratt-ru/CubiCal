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

verbose = 0

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


def compute_residual(resid_arr, obser_arr, model_arr, gains, t_int=1, f_int=1):
    """
    This function computes the residual. This is the difference between the
    observed data, and the model data with the gains applied to it.

    Args:
        resid_arr (np.array): Array which will receive residuals.
                          Shape is n_dir, n_tim, n_fre, n_ant, a_ant, n_cor, n_cor
        obser_arr (np.array): Array containing the observed visibilities.
                          Same shape
        model_arr (np.array): Array containing the model visibilities.
                          Same shape
        gains (np.array): Array containing the current gain estimates.
                          Shape of n_dir, n_timint, n_freint, n_ant, n_cor, n_cor
                          Where n_timint = ceil(n_tim/t_int), n_fre = ceil(n_fre/t_int)

    Returns:
        residual (np.array): Array containing the result of computing D-GMG^H.
    """
    
    gains_h = gains.transpose(0,1,2,3,5,4).conj()

    cyfull.cycompute_residual(model_arr, gains, gains_h, obser_arr, resid_arr, t_int, f_int)

    return resid_arr

def retile_array(in_arr, m1, m2, n1, n2):
    """
    Retiles a 2D array of shape m, n into shape m1, m2, n1, n2. If tiling is perfect, 
    i.e. m1*m2 = m, n1*n2 =n, then this returns a reshaped array. Otherwise, it creates a new 
    array and copies data.
    """
    
    # TODO: Investigate writing a kernel that accomplishes this and the relevant summation.

    m, n = in_arr.shape

    new_shape = (m1, m2, n1, n2)

    if (m1*m2 == m) and (n1*n2 == n):
        
        return in_arr.reshape(new_shape)
    
    else:
        
        out_arr = np.zeros(new_shape, dtype=in_arr.dtype)
        out_arr.reshape((m1*m2, n1*n2))[:m,:n] = in_arr
        
        return out_arr

def estimate_noise(data, flags):
    """
    Given a data cube with dimensions (n_tim, n_fre, n_ant, n_ant, n_cor, n_cor) and a flag cube 
    (n_tim, n_fre, n_ant, n_ant), this function estimates the noise in the data.

    Returns tuple of noise, inverse_noise_per_antenna_squared and inverse_noise_per_channel_squared.
    """

    n_tim, n_fre, n_ant, n_ant, n_cor, n_cor = data.shape
    
    # Create a boolean flag array from the bitflags. Construct delta flags by or-ing flags in
    # channel n with flags in channel n+1.

    deltaflags = (flags!=0)
    deltaflags[:, 1:, ...] |= deltaflags[:, :-1, ...]
    deltaflags[:, 0 , ...]  = deltaflags[:,   1, ...]
    
    # Create array for the squared difference between channel-adjacent visibilities.

    deltavis2 = np.zeros((n_tim, n_fre, n_ant, n_ant), np.float32)
    
    # Square the absolute value of the difference between channel-adjacent visibilities and sum 
    # over correlations. Normalize the result by n_cor*n_cor*4. The factor of 4 arises because 
    # Var<c1-c2> = Var<c1>+Var<c2> and Var<c>=Var<r>+Var<i>. Thus, the square of the abs difference
    # between two complex visibilities has contributions from _four_ noise terms.

    # TODO: When fewer than 4 correlations are provided, the normalisation needs to be different.

    deltavis2[:, 1:, ...]  = np.square(abs(data[:, 1:, ...] - data[:, :-1, ...])).sum(axis=(-2,-1))
    deltavis2[:, 1:, ...] /= n_cor*n_cor*4
    deltavis2[:, 0 , ...]  = deltavis2[:, 1, ...]
    
    # The flagged elements are zeroed; we don't have an adequate noise estimate for those channels.

    deltavis2[deltaflags] = 0
    
    # This flag inversion gives a count of the valid estimates in deltavis2.

    deltaflags = ~deltaflags
    
    # This computes the inverse variance per antenna and per channel as well as an overall noise 
    # estimate. Warnings are supressed as divide by zero is expected.

    with np.errstate(divide='ignore', invalid='ignore'):
        noise_est = math.sqrt(deltavis2.sum() / deltaflags.sum())
        inv_var_ant  = deltaflags.sum(axis=(0, 1, 2)) / deltavis2.sum(axis=(0, 1, 2))
        inv_var_chan = deltaflags.sum(axis=(0, 2, 3)) / deltavis2.sum(axis=(0, 2, 3))
    
    # Elements may have been set_trace to NaN due to division by zero. This simply zeroes those elements.
    
    inv_var_ant[~np.isfinite(inv_var_ant)] = 0
    inv_var_chan[~np.isfinite(inv_var_chan)] = 0
    
    return noise_est, inv_var_ant, inv_var_chan


def solve_gains(obser_arr, model_arr, flags_arr, min_delta_g=1e-6, maxiter=30,
                chi_tol=1e-5, chi_interval=5, t_int=1, f_int=1, label=""):
    """
    This function is the main body of the GN/LM method. It handles iterations
    and convergence tests.

    Args:
        obser_arr (np.array: n_tim, n_fre, n_ant, n_ant, n_cor, n_cor): Array containing the observed visibilities.
        model_arr (np.array: n_dir, n_tim, n_fre, n_ant, n_ant, n_cor, n_cor): Array containing the model visibilities.
        flags_arr (np.array: n_tim, n_fre, n_ant, n_ant): int array flagging invalid points
        min_delta_g (float): Gain improvement threshold.
        maxiter (int): Maximum number of iterations allowed.
        chi_tol (float): Chi-squared improvement threshold (relative)
        chi_interval (int): Interval at which the chi-squared test is performed.

    Returns:
        gains (np.array): Array containing the final gain estimates.
    """

    n_dir, n_tim, n_fre, n_ant, n_ant, n_cor, n_cor = model_arr.shape

    # n_tim and n_fre are the time and frequency dimensions of the data arrays.
    # n_timint and n_freint are the time and frequnecy dimensions of the gains.

    n_timint = int(math.ceil(float(n_tim)/t_int))   # Number of time intervals
    n_freint = int(math.ceil(float(n_fre)/f_int))   # Number of freq intervals
    n_tf  = n_fre*n_tim                             # Number of time-freq slots
    n_int = n_timint*n_freint                       # Number of solution intervals

    # Initialize gains to the appropriate shape with all gains set to identity. Create a copy to 
    # hold the gain of the previous iteration. 

    gain_shape = [n_dir, n_timint, n_freint, n_ant, n_cor, n_cor]
    
    gains     = np.empty(gain_shape, dtype=obser_arr.dtype)
    gains[:]  = np.eye(n_cor) 
    old_gains = gains.copy()

    # Initialize some numbers used in convergence testing.

    n_cnvgd = 0 # Number of converged solutions
    n_stall = 0 # Number of intervals with stalled chi-sq
    n_sols = float(n_dir*n_int) # Number of gains solutions
    n_vis2x2 = n_tf*n_ant*n_ant # Number of 2x2 visbilities
    iters = 0   

    # Estimates the overall noise level and the inverse variance per channel and per antenna as 
    # noise varies across the band. This is used to normalize chi^2.

    noise_est, inv_var_ant, inv_var_chan = estimate_noise(obser_arr, flags_arr)

    # TODO: Check number of equations per solution interval, and deficient flag intervals.
    unflagged = (flags_arr==0)

    # (n_ant) vector containing the number of valid equations per antenna.
    # Factor of two is necessary as we have the conjugate of each equation too.

    eqs_per_antenna = 2*np.sum(unflagged, axis=(0, 1, 2))

    # (n_tim, n_fre) array containing number of valid equations per each time/freq slot.

    eqs_per_tf_slot = np.sum(unflagged, axis=(-1, -2))*n_cor*n_cor*2

    # (n_timint, n_freint) array containing number of valid equations per each time/freq interval.
    
    eqs_per_interval = retile_array(eqs_per_tf_slot, n_timint, t_int, n_freint, f_int).sum(axis=(1,3))

    # The following determines the number of valid (unflagged) time/frequency slots and the number 
    # of valid solution intervals.

    valid_tf_slots  = eqs_per_tf_slot>0
    valid_intervals = eqs_per_interval>0
    num_valid_tf_slots  = valid_tf_slots.sum()
    num_valid_intervals = valid_intervals.sum()

    # In the event that there are no solution intervals with valid data, this will log some of the
    # flag information.

    if num_valid_intervals is 0:
        flagstats = OrderedDict()
        for cat, bitmask in FL.categories().iteritems():
            flagstats[cat] = ((flags_arr&bitmask) != 0).sum()/(n_cor*n_cor)
        makestring = lambda cat,total,n_vis2x2: "%s:%d(%.2f%%)" % (cat, total, total*100./n_vis2x2)
        flagstat_strings = [makestring(cat, total, n_vis2x2) for cat, total in flagstats.iteritems() if total]
        print>> log, "{} no valid solution intervals. Flags are {}".format(label, " ".join(flagstat_strings or ["none"]))
        print>> log, "{}: no valid solution intervals. All data flagged perhaps?".format(label)
        return gains

    mineqs = eqs_per_interval[valid_intervals].min()
    maxeqs = eqs_per_interval.max()
    # compute chi-sq normalization term per each solution interval
    chisq_norm = np.ones_like(eqs_per_interval, dtype=obser_arr.real.dtype)
    with np.errstate(divide='ignore'):  # ignore division by 0
        chisq_norm /= eqs_per_interval
    chisq_norm[~valid_intervals] = 0


    # make tiled residual array (tiled by whole time/freq intervals)
    residual_tiled = np.zeros((n_timint,t_int,n_freint,f_int,n_ant,n_ant,n_cor,n_cor), obser_arr.dtype)
    # make ntim,nfreq view into this array
    residual = residual_tiled.reshape((n_timint*t_int,n_freint*f_int,n_ant,n_ant,n_cor,n_cor))[:n_tim,:n_fre,...]

    # compute initial residual
    compute_residual(residual, obser_arr, model_arr, gains, t_int, f_int)

    # chi^2 is computed by summing over antennas, correlations and intervals. Do time, antennas and corrs first,
    # then normalize by per-channel noise-squared, then collapse freq intervals
    chi = np.sum(np.square(np.abs(residual_tiled)), axis=(1,4,5,6,7))
    # chi is now reduced to n_timint,n_freint,f_int in shape --
    chi.reshape((n_timint,n_freint*f_int))[:,:n_fre] *= inv_var_chan[np.newaxis,:]
    chi = np.sum(chi, axis=2) * chisq_norm
    init_chi = mean_chi = chi.sum() / num_valid_intervals

    if verbose > 0:
        flagstats = OrderedDict()
        for cat, bitmask in FL.categories().iteritems():
            flagstats[cat] = (flags_arr&bitmask != 0).sum()/(n_cor*n_cor)
        flagstat_strings = ["%s:%d(%.2f%%)" % (cat, total, total*100./n_vis2x2) for cat, total in flagstats.iteritems() if total]
        print>> log, "{} initial chisq {:.4}, {}/{} valid intervals (min {}/max {} eqs per int). {}/{} valid antennas. Flags are {}".format(label,
                        init_chi, num_valid_intervals, n_int, mineqs, maxeqs,
                        (eqs_per_antenna!=0).sum(), n_ant,
                        " ".join(flagstat_strings or ["none"]))
        # amax = abs(residual).argmax()
        # print>>log, "max init residual {} at index {} {}".format(abs(residual).max(),amax, np.unravel_index(amax, residual.shape))
        # S = slice(amax-2,amax+2)
        # print obser_arr.flatten()[S]
        # print model_arr.flatten()[S]
        # print residual.flatten()[S]

    min_quorum = 0.99
    warned_null_gain = warned_boom_gain = False

    # TODO: Add better messages.

    while n_cnvgd/n_sols < min_quorum and n_stall/n_int < min_quorum and iters < maxiter:
        iters += 1
        if iters % 2 == 0:
            gains = 0.5*(gains + \
                compute_update(model_arr, obser_arr, gains, t_int, f_int))
        else:
            gains = compute_update(model_arr, obser_arr, gains, t_int, f_int)
        # TODO: various infs and NaNs here indicate something wrong with a solution
        # These should be flagged, and accounted for properly in the statistics
        diff_g = np.sum(np.square(np.abs(old_gains - gains)), axis=(-1,-2,-3))
        norm_g = np.sum(np.square(np.abs(gains)), axis=(-1,-2,-3))
        # diff_g and norm_g have shape of n_dir, n_timint, n_freint; TF slots with no equations
        # will be 0/0, so reset the norm to 1 to avoid division by 0
        norm_g[:,~valid_intervals] = 1
        # any more null Gs? This is unexpected -- report
        null_g = norm_g==0
        if null_g.any():
            norm_g[null_g] = 1
            if not warned_null_gain:
                print>>log, ModColor.Str('{} iteration {} WARNING: {} null gain solution(s) encountered'.format(label, iters, null_g.sum()))
                warned_null_gain = True
        norm_diff_g = diff_g/norm_g
        # count converged solutions. Note that flagged solutions will have a norm_diff_g of 0 by construction

        n_cnvgd = np.sum(norm_diff_g <= min_delta_g**2)

        old_gains = gains.copy()

        if (iters % chi_interval) == 0:
            old_chi, old_mean_chi = chi, mean_chi

            compute_residual(residual, obser_arr, model_arr, gains, t_int, f_int)

            # TODO: some residuals blow up (maybe due to bad data?) and cause np.square() to overflow -- need to flag these
            chi = np.sum(np.square(np.abs(residual_tiled)), axis=(1, 4, 5, 6, 7))
            chi.reshape((n_timint, n_freint * f_int))[:, :n_fre] *= inv_var_chan[np.newaxis, :]
            chi = np.sum(chi, axis=2) * chisq_norm
            mean_chi = chi.sum() / num_valid_intervals

            n_stall = float(np.sum(((old_chi - chi) < chi_tol*old_chi)))
            if verbose > 1:
                print>> log, "{} iteration {} chi-sq is {:.4} delta {:.4}, max gain update {:.4}, converged {:.2%}, stalled {:.2%}".format(label,
                            iters, mean_chi, (old_mean_chi-mean_chi)/old_mean_chi, diff_g.max(), n_cnvgd/n_sols, n_stall/n_int)

    print>>log, "{}: {} iterations done. Converged {:.2%}, stalled {:.2%}. Chisq {:.4} -> {:.4}".format(label,
                iters, n_cnvgd/n_sols, n_stall/n_int, init_chi, mean_chi)

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

def solve_and_save(obser_arr, model_arr, flags_arr, min_delta_g=1e-6, maxiter=30,
                   chi_tol=1e-6, chi_interval=5, t_int=1, f_int=1, label=""):

    gains = solve_gains(obser_arr, model_arr, flags_arr, min_delta_g, maxiter,
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
        parser.add_argument('-delchi','--min_delta_chi', type=float, default=1e-5,
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
        parser.add_argument('-v', '--verbose', type=int, default=0,
                            help='Verbosity level for messages.')

        args = parser.parse_args()

        cPickle.dump(args, open("cubecal.last","w"))

        logger.logToFile(args.log, append=False)
        print>> log, "started: " + " ".join(sys.argv)

    global verbose
    verbose = args.verbose

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
    if args.applyflags:
        ms.apply_flags = True
    if args.bitmask != 0:
        ms.bitmask = args.bitmask

    print>>log, "reading MS columns"
    ms.mass_fetch()
    print>>log, "defining chunks"
    ms.define_chunk(args.tchunk, args.fchunk, single_chunk_id=args.single_chunk_id)

    target = solve_and_save if args.save_corrected else solve_gains

    opts = { "min_delta_g"  : args.min_delta_g,
             "maxiter"      : args.maxiter,
             "chi_tol"      : args.min_delta_chi,
             "chi_interval" : args.chi_interval,
             "t_int"        : args.tint,
             "f_int"        : args.fint }


    t0 = time()

    # debugging mode: run serially (also if nproc not set, or single chunk is specified)
    if debugging or not args.processes or args.single_chunk_id:
        for obser, model, flags, weight, chunk_label in ms:
            if target is solve_and_save:
                gains, covis = target(obser, model, flags, label = chunk_label, **opts)
                ms.arr_to_col(covis, [ms._chunk_ddid, ms._chunk_tchunk, ms._first_f, ms._last_f])
            else:
                gains = target(obser, model, flags, label = chunk_label, **opts)
            ms.add_to_gain_dict(gains, [ms._chunk_ddid, ms._chunk_tchunk, ms._first_f, ms._last_f],
                                args.tint, args.fint)

    # normal mode: use futures to run in parallel
    else:
        with cf.ProcessPoolExecutor(max_workers=args.processes) as executor:
            future_gains = { executor.submit(target, obser, model, flags, label=chunk_label, **opts) :
                             [ms._chunk_ddid, ms._chunk_tchunk, ms._first_f, ms._last_f]
                             for obser, model, flags, weight, chunk_label in ms }

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
