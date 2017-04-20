"""
Implements the solver loop
"""
import numpy as np
from cubecal.tools import logger, ModColor
from ReadModelHandler import FL
from cubecal.machines import complex_2x2_machine
from cubecal.machines import phase_diag_machine

from statistics import SolverStats

log = logger.getLogger("solver")


def retile_array(in_arr, m1, m2, n1, n2):
    """
    Retiles a 2D array of shape m, n, into shape m1, m2, n1, n2. If tiling is perfect,
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



def solve_gains(obser_arr, model_arr, flags_arr, options, label="", compute_residuals=None):
    """
    This function is the main body of the GN/LM method. It handles iterations
    and convergence tests.

    Args:
        obser_arr (np.array: n_tim, n_fre, n_ant, n_ant, n_cor, n_cor): Array containing the observed visibilities.
        model_arr (np.array: n_dir, n_tim, n_fre, n_ant, n_ant, n_cor, n_cor): Array containing the model visibilities.
        flags_arr (np.array: n_tim, n_fre, n_ant, n_ant): int array flagging invalid points

        options: dict of various solver options (see [solution] section in DefaultParset.cfg)

        chunk_key:         tuple of (n_time_chunk, n_freq_chunk) identifying current chunk
        label:             string label identifying current chunk (e.d. "D0T1F2")

        compute_residuals: if set, final residuals will be computed and returned

    Returns:
        gains (np.array): Array containing the final gain estimates,
        resid (np.array): array containing the final residuals (if compute_residuals is set), else None
    """

    min_delta_g  = options["delta-g"]
    maxiter      = options["max-iter"]
    chi_tol      = options["delta-chi"]
    chi_interval = options["chi-int"]

    # init stats
    stats = SolverStats(obser_arr)
    stats.chunk.label = label

    # init gains machine
    gm = complex_2x2_machine.Complex2x2Gains(model_arr, options)
    # gm = phase_diag_machine.PhaseDiagGains(model_arr, options)

    # Initialize some numbers used in convergence testing.

    n_cnvgd = 0 # Number of converged solutions
    n_stall = 0 # Number of intervals with stalled chi-sq
    n_vis2x2 = gm.n_tf*gm.n_ant*gm.n_ant # Number of 2x2 visbilities
    iters = 0

    # Estimates the overall noise level and the inverse variance per channel and per antenna as
    # noise varies across the band. This is used to normalize chi^2.

    stats.chunk.init_noise, inv_var_antchan, inv_var_ant, inv_var_chan = stats.estimate_noise(obser_arr, flags_arr)

    # TODO: Check number of equations per solution interval, and deficient flag intervals.
    unflagged = (flags_arr==0)

    # (n_ant) vector containing the number of valid equations per antenna.
    # Factor of two is necessary as we have the conjugate of each equation too.

    eqs_per_antenna = 2*np.sum(unflagged, axis=(0, 1, 2))

    # (n_tim, n_fre) array containing number of valid equations per each time/freq slot.

    eqs_per_tf_slot = np.sum(unflagged, axis=(-1, -2))*gm.n_cor*gm.n_cor*2

    # (n_timint, n_freint) array containing number of valid equations per each time/freq interval.

    eqs_per_interval = retile_array(eqs_per_tf_slot, gm.n_timint, gm.t_int, gm.n_freint, gm.f_int).sum(axis=(1,3))

    # The following determines the number of valid (unflagged) time/frequency slots and the number
    # of valid solution intervals.

    valid_tf_slots  = eqs_per_tf_slot>0
    valid_intervals = eqs_per_interval>0
    num_valid_tf_slots  = valid_tf_slots.sum()
    stats.chunk.num_intervals = num_valid_intervals = valid_intervals.sum()

    # In the event that there are no solution intervals with valid data, this will log some of the
    # flag information. This also breaks out of the function.

    if num_valid_intervals == 0:  # "is 0" doesn't work because np.sum() is a 0-d array

        fstats = ""

        for flag, mask in FL.categories().iteritems():

            n_flag = np.sum((flags_arr & mask) != 0)
            fstats += ("%s:%d(%.2f%%) " % (flag, n_flag, n_flag*100./n_vis2x2)) if n_flag else ""

        print>> log, "{}: no valid solution intervals. Flags are: \n{}".format(label, fstats)

        return gm, (obser_arr if compute_residuals else None), stats

    # Compute chi-squared normalization factor for each solution interval (used by compute_chisq() below)
    chisq_norm = np.zeros_like(eqs_per_interval, dtype=obser_arr.real.dtype)
    chisq_norm[valid_intervals] = (1./eqs_per_interval[valid_intervals])

    # Initialize a tiled residual array (tiled by whole time/freq intervals). Shapes correspond to
    # tiled array shape and the intermediate shape from which our view of the residual is selected.

    tiled_shape = [gm.n_timint, gm.t_int, gm.n_freint, gm.f_int, gm.n_ant, gm.n_ant, gm.n_cor, gm.n_cor]
    inter_shape = [gm.n_timint*gm.t_int, gm.n_freint*gm.f_int, gm.n_ant, gm.n_ant, gm.n_cor, gm.n_cor]

    tiled_resid_arr = np.zeros(tiled_shape, obser_arr.dtype)
    resid_arr = tiled_resid_arr.reshape(inter_shape)[:gm.n_tim,:gm.n_fre,...]
    gm.compute_residual(obser_arr, model_arr, resid_arr)
    # this flag is set to True when we have an up-to-date residual in resid_arr
    have_residuals = True

    def compute_chisq(statfield=None):
        """
        Computes chi-sq statistic based on current residuals.

        Returns chi,mean_chi, where
            chi is normalized chi-sq per solution interval (shape (n_timint, n_freint))
            mean_chi is single chi-sq value for the entire chunk

        If statfield is given, populates stats arrays with the appropriate sums.
        """
        # Chi-squared is computed by summation over antennas, correlations and intervals. Sum over
        # time intervals, antennas and correlations first. Normalize by per-channel variance and finally
        # sum over frequency intervals.
        # TODO: Some residuals blow up and cause np.square() to overflow -- need to flag these.
        # sum chi-square over correlations, and one antenna axis, to shape n_timint, t_int, n_freint, f_int, n_ant
        chi0    = np.sum(np.square(np.abs(tiled_resid_arr)), axis=(5,6,7))
        # take a view into this array, of shape n_tim, n_fre, n_ant
        chi1    = chi0.reshape((gm.n_timint*gm.t_int, gm.n_freint*gm.f_int, gm.n_ant))[:gm.n_tim, :gm.n_fre, :]
        # number of terms in this chi-square sum, shape n_tim, n_fre, n_ant
        nterms  = 2*gm.n_cor*gm.n_cor*np.sum(unflagged, axis=3)
        # normalize this by the per-channel variance
        chi1 *= inv_var_chan[np.newaxis, :, np.newaxis]
        # now collapse into sum per solution interval, and overall sum
        chi_int = np.sum(chi0, axis=(1,3,4)) * chisq_norm    # chisq_norm is already precomputed as 1/nterms per interval
        chi_tot = np.sum(chi0) / np.sum(eqs_per_interval)
        # if stats are requested, collapse into stat arrays
        if statfield:
            getattr(stats.chanant, statfield)[...]  = np.sum(chi1, axis=0)
            getattr(stats.timeant, statfield)[...]  = np.sum(chi1, axis=1)
            getattr(stats.timechan, statfield)[...] = np.sum(chi1, axis=2)
            getattr(stats.chanant, statfield+'n')[...] = np.sum(nterms, axis=0)
            getattr(stats.timeant, statfield+'n')[...] = np.sum(nterms, axis=1)
            getattr(stats.timechan, statfield+'n')[...] = np.sum(nterms, axis=2)
        return chi_int, chi_tot

    chi, mean_chi = compute_chisq(statfield='initchi2')
    stats.chunk.init_chi2 = mean_chi

    old_gains = gm.gains.copy()

    # The following provides some debugging information when verbose is set to > 0.

    if log.verbosity() > 0:

        mineqs = eqs_per_interval[valid_intervals].min()
        maxeqs = eqs_per_interval.max()
        anteqs = np.sum(eqs_per_antenna!=0)

        fstats = ""

        for flag, mask in FL.categories().iteritems():

            n_flag = np.sum((flags_arr & mask) != 0)/(gm.n_cor*gm.n_cor)
            fstats += ("%s:%d(%.2f%%) " % (flag, n_flag, n_flag*100./n_vis2x2)) if n_flag else ""

        print>> log, ("{} Initial chi2 = {:.4}, {}/{} valid intervals (min {}/max {} eqs per int),"
                      " {}/{} valid antennas, noise {:.3}, flags: {}").format(   label,
                                                                        mean_chi,
                                                                        num_valid_intervals,
                                                                        gm.n_int,
                                                                        mineqs,
                                                                        maxeqs,
                                                                        anteqs,
                                                                        gm.n_ant,
                                                                        float(stats.chunk.init_noise),
                                                                        fstats  )

    min_quorum = 0.99
    warned_null_gain = warned_boom_gain = False

    # Main loop of the NNLS method. Terminates after quorum is reached in either converged or
    # stalled solutions or when the maximum number of iterations is exceeded.

    gm.precompute_attributes(model_arr)

    while n_cnvgd/gm.n_sols < min_quorum and n_stall/gm.n_int < min_quorum and iters < maxiter:

        iters += 1

        gm.compute_update(model_arr, obser_arr, iters)

        # if iters % 2 == 0:
        #     gm.gains = 0.5*(gm.gains + gm.compute_update(model_arr, obser_arr))
        # else:
        #     gm.gains = gm.compute_update(model_arr, obser_arr)

        have_residuals = False

        # TODO: various infs and NaNs here indicate something wrong with a solution. These should
        # be flagged and accounted for properly in the statistics.

        # Compute values used in convergence tests.

        diff_g = np.sum(np.square(np.abs(old_gains - gm.gains)), axis=(-1,-2,-3))
        norm_g = np.sum(np.square(np.abs(gm.gains)), axis=(-1,-2,-3))
        norm_g[:,~valid_intervals] = 1      # Prevents division by zero.

        # Checks for unexpected null gain solutions and logs a warning.

        null_g = (norm_g==0)

        if null_g.any():
            norm_g[null_g] = 1
            if not warned_null_gain:
                print>>log, ModColor.Str("{} iter {} WARNING: {} null gain solution(s) "
                                         "encountered".format(label, iters, null_g.sum()))
                warned_null_gain = True

        # Count converged solutions based on norm_diff_g. Flagged solutions will have a norm_diff_g
        # of 0 by construction.

        norm_diff_g = diff_g/norm_g
        n_cnvgd = np.sum(norm_diff_g <= min_delta_g**2)

        # Update old gains for subsequent convergence tests.

        old_gains = gm.gains.copy()

        # Check residual behaviour after a number of iterations equal to chi_interval. This is
        # expensive, so we do it as infrequently as possible.

        if (iters % chi_interval) == 0:

            old_chi, old_mean_chi = chi, mean_chi

            gm.compute_residual(obser_arr, model_arr, resid_arr)
            chi, mean_chi = compute_chisq()
            have_residuals = True

            # Check for stalled solutions - solutions for which the residual is no longer improving.

            n_stall = float(np.sum(((old_chi - chi) < chi_tol*old_chi)))

            if log.verbosity() > 1:

                delta_chi = (old_mean_chi-mean_chi)/old_mean_chi

                print>> log, ("{} iter {} chi2 {:.4} delta {:.4}, max gain update {:.4}, "
                              "conv {:.2%}, stalled {:.2%}").format(   label,
                                                                            iters,
                                                                            mean_chi,
                                                                            delta_chi,
                                                                            diff_g.max(),
                                                                            n_cnvgd/gm.n_sols,
                                                                            n_stall/gm.n_int   )

    # do we need to recompute the final residuals?
    if (options['last-rites'] or compute_residuals) and not have_residuals:
        gm.compute_residual(obser_arr, model_arr, resid_arr)
        if options['last-rites']:
            # recompute chi^2 based on original noise statistics
            chi, mean_chi = compute_chisq(statfield='chi2')

    # re-estimate the noise using the final residuals, if last rites are needed
    if options['last-rites']:
        stats.chunk.noise, inv_var_antchan, inv_var_ant, inv_var_chan = stats.estimate_noise(resid_arr, flags_arr, residuals=True)
        chi1, mean_chi1 = compute_chisq(statfield='chi2')

    stats.chunk.iters = iters
    stats.chunk.num_converged = n_cnvgd
    stats.chunk.num_stalled = n_stall
    stats.chunk.chi2 = mean_chi

    message = ("{}: {} iters, conv {:.2%}, stalled {:.2%}, "
                "chi2 {:.4} -> {:.4}").format(label,
                                                iters,
                                                n_cnvgd/gm.n_sols,
                                                n_stall/gm.n_int,
                                                float(stats.chunk.init_chi2),
                                                mean_chi)
    if options['last-rites']:
        message += " ({:.4}), noise {:.3} -> {:.3}".format(float(mean_chi1), float(stats.chunk.init_noise), float(stats.chunk.noise))

    print>>log, message


    return gm, (resid_arr if compute_residuals else None), stats



def solve_and_correct(obser_arr, model_arr, flags_arr, weight_arr, options, label=""):
    # if weights are set, multiply data and model by weights, but keep an unweighted copy
    # of the data, since we need to correct
    if weight_arr is not None:
        obser_arr1 = obser_arr*weight_arr[..., np.newaxis, np.newaxis]
        model_arr *= weight_arr[np.newaxis, ..., np.newaxis, np.newaxis]
    else:
        obser_arr1 = obser_arr

    gm, _, stats = solve_gains(obser_arr1, model_arr, flags_arr, options, label=label)

    corr_vis = gm.apply_inv_gains(obser_arr)

    return gm, corr_vis, stats


def solve_and_correct_res(obser_arr, model_arr, flags_arr, weight_arr, options, label=""):
    # if weights are set, multiply data and model by weights, but keep an unweighted copy
    # of the model and data, since we need to correct the residuals

    if weight_arr is not None:
        obser_arr1 = obser_arr * weight_arr[..., np.newaxis, np.newaxis]
        model_arr1 = model_arr * weight_arr[np.newaxis, ..., np.newaxis, np.newaxis]
    else:
        obser_arr1, model_arr1 = obser_arr, model_arr

    # use the residuals computed in solve_gains() only if no weights. Otherwise need
    # to recompute them from unweighted versions
    gm, resid_arr, stats = solve_gains(obser_arr, model_arr, flags_arr, options, label=label,
                                       compute_residuals=(weight_arr is None))

    if weight_arr is not None:
        gm.compute_residual(obser_arr, model_arr, resid_arr)

    corr_vis = gm.apply_inv_gains(resid_arr)

    return gm, corr_vis, stats
