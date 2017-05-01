"""
Implements the solver loop
"""
import numpy as np
import traceback
from cubecal.tools import logger, ModColor
from data_handler import FL, Tile
from cubecal.tools import shared_dict
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



def _solve_gains(obser_arr, model_arr, flags_arr, options, label="", compute_residuals=None):
    """
    This function is the main body of the GN/LM method. It handles iterations
    and convergence tests.

    Args:
        obser_arr (np.array: n_mod, n_tim, n_fre, n_ant, n_ant, n_cor, n_cor): Array containing the observed visibilities.
        model_arr (np.array: n_dir, n_mod, n_tim, n_fre, n_ant, n_ant, n_cor, n_cor): Array containing the model visibilities.
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
    min_ampl     = options["clip-low"]
    max_ampl     = options["clip-high"]
    clip_after_iter = options["clip-after-iter"]

    # init stats
    stats = SolverStats(obser_arr)
    stats.chunk.label = label

    # init gains machine
    if options['jones-type'] == 'complex-2x2':
        gm = complex_2x2_machine.Complex2x2Gains(model_arr, options)
    elif options['jones-type'] == 'phase-diag':
        gm = phase_diag_machine.PhaseDiagGains(model_arr, options)
    else:
        raise ValueError("unknown jones-type '{}'".format(options['jones-type']))

    # Initialize some numbers used in convergence testing.

    n_cnvgd = 0 # Number of converged solutions
    n_stall = 0 # Number of intervals with stalled chi-sq
    n_vis2x2 = gm.n_tf*gm.n_ant*gm.n_ant # Number of 2x2 visbilities
    iters = 0

    # Estimates the overall noise level and the inverse variance per channel and per antenna as
    # noise varies across the band. This is used to normalize chi^2.

    stats.chunk.init_noise, inv_var_antchan, inv_var_ant, inv_var_chan = stats.estimate_noise(obser_arr, flags_arr)

    def compute_stats(flags, statfields):
        """Helper function: recomputes various stats and totals based on the current state of the flags.
        Needed because flags may change during iteration (due to solution flagging), so we may need
        to redo it"""
        unflagged = (flags==0)

        # (n_ant) vector containing the number of valid equations per antenna.
        # Factor of two is necessary as we have the conjugate of each equation too.
        eqs_per_antenna = 2*np.sum(unflagged, axis=(0, 1, 2))*gm.n_mod

        # (n_tim, n_fre) array containing number of valid equations per each time/freq slot.
        eqs_per_tf_slot = np.sum(unflagged, axis=(-1, -2))*gm.n_mod*gm.n_cor*gm.n_cor*2

        # (n_timint, n_freint) array containing number of valid equations per each time/freq interval.
        eqs_per_interval = retile_array(eqs_per_tf_slot, gm.n_timint, gm.t_int, gm.n_freint, gm.f_int).sum(axis=(1,3))

        # The following determines the number of valid (unflagged) time/frequency slots and the number
        # of valid solution intervals.
        valid_intervals = eqs_per_interval>0
        stats.chunk.num_intervals = num_valid_intervals = valid_intervals.sum()

        # Compute chi-squared normalization factor for each solution interval (used by compute_chisq() below)
        chisq_norm = np.zeros_like(eqs_per_interval, dtype=obser_arr.real.dtype)
        chisq_norm[valid_intervals] = (1. / eqs_per_interval[valid_intervals])

        if statfields:
            # compute number of terms in each chi-square sum, shape is n_tim, n_fre, n_ant
            nterms  = 2*gm.n_cor*gm.n_cor*np.sum(unflagged, axis=3)
            # update the stats structure accordingly
            for field in statfields:
                getattr(stats.chanant,  field+'n')[...] = np.sum(nterms, axis=0)
                getattr(stats.timeant,  field+'n')[...] = np.sum(nterms, axis=1)
                getattr(stats.timechan, field+'n')[...] = np.sum(nterms, axis=2)

        return eqs_per_antenna, eqs_per_interval, valid_intervals, num_valid_intervals, chisq_norm

    # compute initial stats
    eqs_per_antenna, eqs_per_interval, valid_intervals, num_valid_intervals, chisq_norm = \
        compute_stats(flags_arr, statfields=('initchi2', 'chi2'))

    # In the event that there are no solution intervals with valid data, this will log some of the
    # flag information. This also breaks out of the function.

    if num_valid_intervals == 0:  # "is 0" doesn't work because np.sum() is a 0-d array

        fstats = ""

        for flag, mask in FL.categories().iteritems():

            n_flag = np.sum((flags_arr & mask) != 0)
            fstats += ("%s:%d(%.2f%%) " % (flag, n_flag, n_flag*100./n_vis2x2)) if n_flag else ""

        print>> log, "{} is completely flagged: {}".format(label, fstats)

        return gm, (obser_arr if compute_residuals else None), stats


    # Initialize a tiled residual array (tiled by whole time/freq intervals). Shapes correspond to
    # tiled array shape and the intermediate shape from which our view of the residual is selected.

    tiled_shape = [gm.n_mod, gm.n_timint, gm.t_int, gm.n_freint, gm.f_int, gm.n_ant, gm.n_ant, gm.n_cor, gm.n_cor]
    inter_shape = [gm.n_mod, gm.n_timint*gm.t_int, gm.n_freint*gm.f_int, gm.n_ant, gm.n_ant, gm.n_cor, gm.n_cor]

    tiled_resid_arr = np.zeros(tiled_shape, obser_arr.dtype)
    resid_arr = tiled_resid_arr.reshape(inter_shape)[:, :gm.n_tim, :gm.n_fre, ...]
    gm.compute_residual(obser_arr, model_arr, resid_arr)
    # this flag is set to True when we have an up-to-date residual in resid_arr
    have_residuals = True

    # same thing for flags, we need to tile the flags into time/freq intervals?
    # If needed, make a separate retiled array and copy flags to it. Else use original array.
    # Preserve original array for later, as we need to copy any flags out to it

    tiled_flag_shape = tiled_shape[1:-2]
    flags_arr_orig = flags_arr
    if tiled_resid_arr.size == resid_arr.size:
        tiled_flags_arr = flags_arr.reshape(tiled_flag_shape)
    else:
        inter_flag_shape = inter_shape[1:-2]
        tiled_flags_arr = np.full(tiled_flag_shape, FL.MISSING, flags_arr.dtype)
        flags_arr = tiled_flags_arr.reshape(inter_flag_shape)[:gm.n_tim, :gm.n_fre, ...]
        flags_arr[:] = flags_arr_orig[:]

    # pre-flag gain solution intervals that are completely flagged in the input data (i.e. MISSING|PRIOR)
    # this has shape n_timint, n_freint, n_ant
    missing_gains = (tiled_flags_arr&(FL.MISSING|FL.PRIOR) != 0).all(axis=(1,3,5))

    # gain flags have shape n_dir, n_timint, n_freint, n_ant
    # all intervals with no prior data are flagged as FL.MISSING
    gm.gflags[:, missing_gains] = FL.MISSING
    missing_gain_fraction = missing_gains.sum() / float(missing_gains.size)

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
        # sum chi-square over correlations, models, and one antenna axis, to shape n_timint, t_int, n_freint, f_int, n_ant
        chi0    = np.sum(np.square(np.abs(tiled_resid_arr)), axis=(0,6,7,8))
        # take a view into this array, of shape n_tim, n_fre, n_ant
        chi1    = chi0.reshape((gm.n_timint*gm.t_int, gm.n_freint*gm.f_int, gm.n_ant))[:gm.n_tim, :gm.n_fre, :]
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
    raised_new_gain_flags = False
    num_gain_flags = (gm.gflags&~FL.MISSING != 0).sum()

    # Main loop of the NNLS method. Terminates after quorum is reached in either converged or
    # stalled solutions or when the maximum number of iterations is exceeded.

    gm.precompute_attributes(model_arr)

    while n_cnvgd/gm.n_sols < min_quorum and n_stall/gm.n_int < min_quorum and iters < maxiter:

        iters += 1

        gm.compute_update(model_arr, obser_arr, iters)
        gabs = abs(gm.gains)
        # flag solution on various pathologies

        # anything previously flagged for another reason will not be flagged again
        gflagged = gm.gflags != 0

        # a gain solution has gone to inf/nan
        boom = (~np.isfinite(gm.gains)).any(axis=(-1,-2))  # collapse correlation axes
        gm.gflags[boom&~gflagged] |= FL.BOOM
        gflagged |= boom

        # a diagonal gain solution has gone to 0
        gnull = (gm.gains[..., 0, 0] == 0) | (gm.gains[..., 1, 1] == 0)
        # excepting those which was at zero for other reasons
        gm.gflags[gnull&~gflagged] |= FL.GNULL
        gflagged |= gnull

        # a gain solution is out of bounds
        if iters >= clip_after_iter and max_ampl or min_ampl:
            goob = np.zeros(gabs.shape, bool)
            if max_ampl:
                goob = gabs.max(axis=(-1, -2)) > max_ampl
            if min_ampl:
                goob |= (gabs[...,0,0]<min_ampl) | (gabs[...,1,1,]<min_ampl)
            gm.gflags[goob&~gflagged] |= FL.GOOB
            gflagged |= goob

        # count the gain flags (excepting those set a priori due to missing data)
        nfl = (gm.gflags&~FL.MISSING != 0).sum()

        if nfl > num_gain_flags:
            num_gain_flags = nfl
            # Propagate gain flags out to data
            # Gain flags are per-direction -- data flags are per visibility.
            # For now, just flag everything if _any_ direction is flagged.
            # Note that we take the FL.MISSING bit off when propagating. This is because this bit in the
            # gainflags is pre-set, if data was flagged with PRIOR|MISSING, so we don't want to push that
            # bit back into the data (otherwise every PRIOR but not MISSING flag will become MISSING).
            # if *all* directions are flagged? This would effectively exclude "bad" models from the soluton
            gflags = np.bitwise_or.reduce(gm.gflags, axis=0)  # gflags shape: n_timint, n_freqint, n_ant
            tiled_flags_arr |= gflags[:,np.newaxis,:,np.newaxis,:,np.newaxis]&~FL.MISSING
            tiled_flags_arr |= gflags[:,np.newaxis,:,np.newaxis,np.newaxis,:]&~FL.MISSING
            # recompute various stats
            eqs_per_antenna, eqs_per_interval, valid_intervals, num_valid_intervals, chisq_norm = \
                compute_stats(flags_arr, ('chi2',))
            # re-zero model and data at newly flagged points. TODO: is this needed?
            # TODO: should we perhaps just zero the model per flagged direction, and only flag the data?
            flagged = flags_arr&~(FL.MISSING|FL.PRIOR) !=0
            model_arr[:, :, flagged, :, :] = 0
            obser_arr[   :, flagged, :, :] = 0

            # break out if we find ourseles with no valid solution intervals
            if num_valid_intervals == 0:
                break

        have_residuals = False

        # Compute values used in convergence tests.
        # Note that the check here implicitly marks flagged gains as converged.
        diff_g = np.square(np.abs(old_gains - gm.gains))
        diff_g[gflagged] = 0
        diff_g = diff_g.sum(axis=(-1,-2,-3))
        norm_g = np.square(gabs)
        norm_g[gflagged] = 1
        norm_g = norm_g.sum(axis=(-1,-2,-3))

        norm_diff_g = diff_g/norm_g
        n_cnvgd = (norm_diff_g <= min_delta_g**2).sum()

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
                              "conv {:.2%}, stall {:.2%}, g/fl {:.2%}, d/fl {:2}%").format(label,
                                iters,
                                mean_chi,
                                delta_chi,
                                diff_g.max(),
                                n_cnvgd/gm.n_sols,
                                n_stall/gm.n_int,
                                num_gain_flags/float(gm.gflags.size),
                                missing_gain_fraction)

    # num_valid_intervals will go to 0 if all solution intervals got flagged
    # if not, generate residuals et al.
    if num_valid_intervals:
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

        stats.chunk.chi2 = mean_chi

        message = ("{}: {} iters, conv {:.2%}, stall {:.2%}, g/fl {:.2%}, d/fl {:.2%}, "
                    "chi2 {:.4} -> {:.4}").format(label,
                                                    iters,
                                                    n_cnvgd/gm.n_sols,
                                                    n_stall/gm.n_int,
                                                    num_gain_flags / float(gm.gflags.size),
                                                    missing_gain_fraction,
                                                    float(stats.chunk.init_chi2),
                                                    mean_chi)
        if options['last-rites']:
            message += " ({:.4}), noise {:.3} -> {:.3}".format(float(mean_chi1), float(stats.chunk.init_noise), float(stats.chunk.noise))
        print>> log, message

    # else, everything got flagged, so no valid solutions generated
    else:
        print>>log, ModColor.Str("{} completely flagged after {} iters: g/fl {:.2%}, d/fl {:.2%}").format(label,
                                                    iters,
                                                    num_gain_flags / float(gm.gflags.size),
                                                    missing_gain_fraction)
        stats.chunk.chi2 = 0
        resid_arr = obser_arr

    stats.chunk.iters = iters
    stats.chunk.num_converged = n_cnvgd
    stats.chunk.num_stalled = n_stall

    # copy out flags, if we raised any
    stats.chunk.num_sol_flagged = num_gain_flags
    if num_gain_flags:
        flags_arr_orig[:] = flags_arr
        # also for up message with flagging stats
        fstats = ""
        for flagname, mask in FL.categories().iteritems():
            if mask != FL.MISSING:
                n_flag = (gm.gflags&mask != 0).sum()
                if n_flag:
                    fstats += "{}:{}({:.2%}) ".format(flagname, n_flag, n_flag/float(gm.gflags.size))
        print>> log, ModColor.Str("{} solver flags raised: {}".format(label, fstats))

    return gm, (resid_arr if compute_residuals else None), stats



def solve_only(obser_arr, model_arr, flags_arr, weight_arr, tile, key, label, options):
    # apply weights
    if weight_arr is not None:
        obser_arr *= weight_arr[..., np.newaxis, np.newaxis]
        model_arr *= weight_arr[np.newaxis, ..., np.newaxis, np.newaxis]

    gm, _, stats = _solve_gains(obser_arr, model_arr, flags_arr, options, label=label)

    return gm, None, stats


def solve_and_correct(obser_arr, model_arr, flags_arr, weight_arr, tile, key, label, options):
    # if weights are set, multiply data and model by weights, but keep an unweighted copy
    # of the data, since we need to correct
    if weight_arr is not None:
        obser_arr1 = obser_arr*weight_arr[..., np.newaxis, np.newaxis]
        model_arr *= weight_arr[np.newaxis, ..., np.newaxis, np.newaxis]
    else:
        obser_arr1 = obser_arr

    gm, _, stats = _solve_gains(obser_arr1, model_arr, flags_arr, options, label=label)

    # for corrected visibilities, take the first data/model pair only
    corr_vis = np.zeros_like(obser_arr[0,...])
    gm.apply_inv_gains(obser_arr[0,...], corr_vis)

    return gm, corr_vis, stats


def solve_and_correct_res(obser_arr, model_arr, flags_arr, weight_arr, tile, key, label, options):
    # if weights are set, multiply data and model by weights, but keep an unweighted copy
    # of the model and data, since we need to correct the residuals

    if weight_arr is not None:
        obser_arr1 = obser_arr * weight_arr[..., np.newaxis, np.newaxis]
        model_arr1 = model_arr * weight_arr[np.newaxis, ..., np.newaxis, np.newaxis]
    else:
        obser_arr1, model_arr1 = obser_arr, model_arr

    # use the residuals computed in solve_gains() only if no weights. Otherwise need
    # to recompute them from unweighted versions
    gm, resid_vis, stats = _solve_gains(obser_arr1, model_arr1, flags_arr, options, label=label,
                                        compute_residuals=(weight_arr is None))

    # if we reweighted things above, then recompute the residuals, else use returned residuals
    # note that here we take the first data/model pair only (hence the 0:1 slice)

    if weight_arr is not None:
        resid_vis = np.zeros_like(obser_arr[0:1,...])
        gm.compute_residual(obser_arr[0:1,...], model_arr[:,0:1,...], resid_vis)

    resid_vis = resid_vis[0,...]

    corr_vis = np.zeros_like(resid_vis)
    gm.apply_inv_gains(resid_vis, corr_vis)

    return gm, corr_vis, stats


SOLVERS = { 'solve':          solve_only,
            'solve-correct':  solve_and_correct,
            'solve-residual': solve_and_correct_res
        }


def run_solver(solver_type, itile, chunk_key, options):
    label = None
    try:
        tile = Tile.tile_list[itile]
        label = tile.get_chunk_label(chunk_key)
        solver = SOLVERS[solver_type]

        # invoke solver with cubes from tile

        obser_arr, model_arr, flags_arr, weight_arr = tile.get_chunk_cubes(chunk_key)

        gm, corr_vis, stats = solver(obser_arr, model_arr, flags_arr, weight_arr, tile, chunk_key, label, options)

        # copy results back into tile

        tile.set_chunk_cubes(corr_vis, flags_arr if stats.chunk.num_sol_flagged else None, chunk_key)

        tile.set_chunk_gains(gm.gains, chunk_key)

        return stats
    except Exception, exc:
        print>>log,ModColor.Str("Solver for tile {} chunk {} failed with exception: {}".format(itile, label, exc))
        print>>log,traceback.format_exc()
        raise

