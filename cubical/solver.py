"""
Implements the solver loop
"""
import numpy as np
import traceback
from cubical.tools import logger, ModColor
from cubical.data_handler import FL, Tile
from cubical.tools import shared_dict
from cubical.machines import complex_2x2_machine
from cubical.machines import complex_W_2x2_machine
from cubical.machines import phase_diag_machine
from cubical.statistics import SolverStats

log = logger.getLogger("solver")


def _solve_gains(obser_arr, model_arr, flags_arr, chunk_ts, chunk_fs, options, label="", compute_residuals=None):
    """
    This function is the main body of the GN/LM method. It handles iterations
    and convergence tests.

    Args:
        obser_arr (np.array: n_mod, n_tim, n_fre, n_ant, n_ant, n_cor, n_cor): 
            Array containing the observed visibilities.
        model_arr (np.array: n_dir, n_mod, n_tim, n_fre, n_ant, n_ant, n_cor, n_cor): 
            Array containing the model visibilities.
        flags_arr (np.array: n_tim, n_fre, n_ant, n_ant): 
            Integer array containing flagging data.

        options: 
            Dictionary of various solver options (see [solution] section in DefaultParset.cfg)

        chunk_key:         
            Tuple of (n_time_chunk, n_freq_chunk) which identifies the current chunk.
        label:             
            String label identifying the current chunk (e.d. "D0T1F2").

        compute_residuals: 
            If set, the final residuals will be computed and returned.

    Returns:
        gains (np.array): 
            Array containing the final gain estimates,
        resid (np.array): 
            Array containing the final residuals (if compute_residuals is set), else None.
    """

    min_delta_g  = options["delta-g"]
    maxiter      = options["max-iter"]
    chi_tol      = options["delta-chi"]
    chi_interval = options["chi-int"]
    clip_after_iter = options["clip-after-iter"]

    # Initialise stat object.

    stats = SolverStats(obser_arr)
    stats.chunk.label = label

    # Initialise the chosen gain machine.

    if options['jones-type'] == 'complex-2x2':
        gm = complex_2x2_machine.Complex2x2Gains(model_arr, chunk_ts, chunk_fs, options)
    elif options['jones-type'] == 'phase-diag':
        gm = phase_diag_machine.PhaseDiagGains(model_arr, chunk_ts, chunk_fs, options)
    elif options['jones-type'] == 'robust-2x2':
        gm = complex_W_2x2_machine.ComplexW2x2Gains(model_arr, chunk_ts, chunk_fs, label, options)
    else:
        raise ValueError("unknown jones-type '{}'".format(options['jones-type']))

    iters = 0
    n_stall = 0
    n_tf_slots = gm.n_tim * gm.n_fre

    # Estimates the overall noise level and the inverse variance per channel and per antenna as
    # noise varies across the band. This is used to normalize chi^2.

    stats.chunk.init_noise, inv_var_antchan, inv_var_ant, inv_var_chan = \
                                                        stats.estimate_noise(obser_arr, flags_arr)
    
    # Compute number of equations for the general case.

    def compute_num_eqs(flags, statfields):
        """
        This function computes various stats and totals based on the current state of the flags.
        These values are used for weighting the chi-squared and doing intelligent convergence
        testing.
        """

        unflagged = (flags==0)

        # (n_ant) vector containing the number of valid equations per antenna.
        # Factor of two is necessary as we have the conjugate of each equation too.

        eqs_per_antenna = 2 * np.sum(unflagged, axis=(0, 1, 2)) * gm.n_mod

        # (n_tim, n_fre) array containing number of valid equations for each time/freq slot.
        
        eqs_per_tf_slot = np.sum(unflagged, axis=(-1, -2)) * gm.n_mod * gm.n_cor * gm.n_cor * 2

        if statfields:

            # Compute number of terms in each chi-square sum. Shape is (n_tim, n_fre, n_ant).
            
            nterms  = 2 * gm.n_cor * gm.n_cor * np.sum(unflagged, axis=3)
            
            # Update stats object accordingly.
            
            for field in statfields:
                getattr(stats.chanant,  field+'n')[...] = np.sum(nterms, axis=0)
                getattr(stats.timeant,  field+'n')[...] = np.sum(nterms, axis=1)
                getattr(stats.timechan, field+'n')[...] = np.sum(nterms, axis=2)
    
        return eqs_per_antenna, eqs_per_tf_slot

    eqs_per_antenna, eqs_per_tf_slot = compute_num_eqs(flags_arr, ('initchi2', 'chi2'))

    gm.update_stats(flags_arr, eqs_per_tf_slot)

    # In the event that there are no solution intervals with valid data, this will log some of the
    # flag information and break out of the function.

    if gm.num_valid_intervals == 0: 

        fstats = ""

        for flag, mask in FL.categories().iteritems():
            n_vis2x2 = 100. #TODO remove this when n_vis2x2 is fixed

            n_flag = np.sum((flags_arr & mask) != 0)
            fstats += ("%s:%d(%.2f%%) " % (flag, n_flag, n_flag*100./n_vis2x2)) if n_flag else ""

        print>> log, "{} is completely flagged: {}".format(label, fstats)

        return gm, (obser_arr if compute_residuals else None), stats

    # Initialize a residual array.

    resid_shape = [gm.n_mod, gm.n_tim, gm.n_fre, gm.n_ant, gm.n_ant, gm.n_cor, gm.n_cor]

    resid_arr = np.zeros(resid_shape, obser_arr.dtype)
    gm.compute_residual(obser_arr, model_arr, resid_arr)
    
    # This flag is set to True when we have an up-to-date residual in resid_arr.
    
    have_residuals = True

    def compute_chisq(statfield=None):
        """
        Computes chi-squared statistic based on current residuals.

        Returns chisq_per_tf_slot, chisq_tot, where
            chisq_per_tf_slot is normalized chi-suaredq per time-frequency slot, (n_tim, n_fre).
            chisq_tot is a single chi-squared value for the entire chunk

        If statfield is given, populates stats arrays with the appropriate sums.
        """

        # Chi-squared is computed by summation over antennas, correlations and intervals. Sum over
        # time intervals, antennas and correlations first. Normalize by per-channel variance and 
        # finally sum over frequency intervals.

        # TODO: Some residuals blow up and cause np.square() to overflow -- need to flag these.

        # Sum chi-square over correlations, models, and one antenna axis. Result has shape
        # (n_tim, n_fre, n_ant). We avoid using np.abs by taking a view of the underlying memory.
        # This is substantially faster.

        chisq = np.sum(np.square(resid_arr.view(dtype=resid_arr.real.dtype)), axis=(0,4,5,6))

        # Normalize this by the per-channel variance.

        chisq *= inv_var_chan[np.newaxis, :, np.newaxis]
        
        # Collapse chisq to chi-squared per time-frequency slot and overall chi-squared. norm_factor 
        # is computed as 1/eqs_per_tf_slot.

        norm_factor = np.where(eqs_per_tf_slot>0, 1./eqs_per_tf_slot, 0)

        chisq_per_tf_slot = np.sum(chisq, axis=-1) * norm_factor 

        chisq_tot = np.sum(chisq) / np.sum(eqs_per_tf_slot)

        # If stats are requested, collapse chisq into stat arrays.
        
        if statfield:
            getattr(stats.chanant, statfield)[...]  = np.sum(chisq, axis=0)
            getattr(stats.timeant, statfield)[...]  = np.sum(chisq, axis=1)
            getattr(stats.timechan, statfield)[...] = np.sum(chisq, axis=2)
        
        return chisq_per_tf_slot, chisq_tot

    chi, mean_chi = compute_chisq(statfield='initchi2')
    stats.chunk.init_chi2 = mean_chi

    # The following provides some debugging information when verbose is set to > 0.

    if log.verbosity() > 0:

        mineqs = gm.eqs_per_interval[gm.valid_intervals].min()
        maxeqs = gm.eqs_per_interval.max()
        anteqs = np.sum(eqs_per_antenna!=0)

        n_2x2vis = gm.n_tim * gm.n_fre * gm.n_ant * gm.n_ant

        fstats = ""

        for flag, mask in FL.categories().iteritems():

            n_flag = np.sum((flags_arr & mask) != 0)/(gm.n_cor*gm.n_cor)
            fstats += ("%s:%d(%.2f%%) " % (flag, n_flag, n_flag*100./n_2x2vis)) if n_flag else ""

        logvars = (label, mean_chi, gm.num_valid_intervals, gm.n_tf_ints, mineqs, maxeqs, anteqs, 
                   gm.n_ant, float(stats.chunk.init_noise), fstats)

        print>> log, ("{} Initial chi2 = {:.4}, {}/{} valid intervals (min {}/max {} eqs per int),"
                      " {}/{} valid antennas, noise {:.3}, flags: {}").format(*logvars)

    min_quorum = 0.99
    n_gflags = (gm.gflags&~FL.MISSING != 0).sum()

    # Do any precomputation required by the current gain machine.

    gm.precompute_attributes(model_arr)

    # Main loop of the NNLS method. Terminates after quorum is reached in either converged or
    # stalled solutions or when the maximum number of iterations is exceeded.

    while gm.n_cnvgd/gm.n_sols < min_quorum and n_stall/n_tf_slots < min_quorum and iters < maxiter:

        iters += 1

        gm.compute_update(model_arr, obser_arr, iters)
        
        gm.flag_solutions(iters>clip_after_iter)

        # If the number of flags had increased, these need to be propagated out to the data. Note
        # that gain flags are per-direction whereas data flags are per visibility. Currently, just
        # everything is flagged if any direction is flagged.

        # We remove the FL.MISSING bit when propagating as this bit is pre-set for data was flagged 
        # as PRIOR|MISSING. This prevents every PRIOR but not MISSING flag from becoming MISSING.

        if gm.n_flagged > n_gflags:
            
            n_gflags = gm.n_flagged
            
            gm.propagate_gflags(flags_arr)

            # Recompute various stats now that the flags raised by the gain machine have been 
            # propagated into the flags_arr.
            
            eqs_per_antenna, eqs_per_tf_slot = compute_num_eqs(flags_arr, ('chi2',))

            gm.update_stats(flags_arr, eqs_per_tf_slot)

            # Re-zero the model and data at newly flagged points. TODO: is this needed?
            # TODO: should we perhaps just zero the model per flagged direction, and only flag the data?
            
            new_flags = flags_arr&~(FL.MISSING|FL.PRIOR) !=0
            model_arr[:, :, new_flags, :, :] = 0
            obser_arr[   :, new_flags, :, :] = 0

            # Break out of the solver loop if we find ourselves with no valid solution intervals.
            
            if gm.num_valid_intervals == 0:
                break

        have_residuals = False

        # Compute values used in convergence tests. This check implicitly marks flagged gains as 
        # converged.
        
        gm.update_conv_params(min_delta_g)

        # Update old gains for subsequent convergence tests.

        gm.old_gains = gm.gains.copy()

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

                logvars = (label, iters, mean_chi, delta_chi, gm.max_update, gm.n_cnvgd/gm.n_sols,
                           n_stall/n_tf_slots, n_gflags/float(gm.gflags.size),
                           gm.missing_gain_fraction)

                print>> log, ("{} iter {} chi2 {:.4} delta {:.4}, max gain update {:.4}, "
                              "conv {:.2%}, stall {:.2%}, g/fl {:.2%}, d/fl {:2}%").format(*logvars)

    # num_valid_intervals will go to 0 if all solution intervals were flagged. If this is not the 
    # case, generate residuals etc.
    
    if gm.num_valid_intervals:

        # Do we need to recompute the final residuals?
        if (options['last-rites'] or compute_residuals) and not have_residuals:
            gm.compute_residual(obser_arr, model_arr, resid_arr)
            if options['last-rites']:
                # Recompute chi-squared based on original noise statistics.
                chi, mean_chi = compute_chisq(statfield='chi2')

        # Re-estimate the noise using the final residuals, if last rites are needed.

        if options['last-rites']:
            stats.chunk.noise, inv_var_antchan, inv_var_ant, inv_var_chan = \
                                        stats.estimate_noise(resid_arr, flags_arr, residuals=True)
            chi1, mean_chi1 = compute_chisq(statfield='chi2')

        stats.chunk.chi2 = mean_chi

        logvars = (label, iters, gm.n_cnvgd/gm.n_sols, n_stall/n_tf_slots, 
                   n_gflags / float(gm.gflags.size), gm.missing_gain_fraction,
                   float(stats.chunk.init_chi2), mean_chi)

        message = ("{}: {} iters, conv {:.2%}, stall {:.2%}, g/fl {:.2%}, d/fl {:.2%}, "
                    "chi2 {:.4} -> {:.4}").format(*logvars)

        if options['last-rites']:

            logvars = (float(mean_chi1), float(stats.chunk.init_noise), float(stats.chunk.noise))

            message += " ({:.4}), noise {:.3} -> {:.3}".format(*logvars)
        
        print>> log, message

    # If everything has been flagged, no valid solutions are generated. 

    else:
        
        logvars = (label, iters, n_gflags / float(gm.gflags.size), gm.missing_gain_fraction)

        print>>log, ModColor.Str("{} completely flagged after {} iters:"
                                 " g/fl {:.2%}, d/fl {:.2%}").format(*logvars)
        
        stats.chunk.chi2 = 0
        resid_arr = obser_arr

    stats.chunk.iters = iters
    stats.chunk.num_converged = gm.n_cnvgd
    stats.chunk.num_stalled = n_stall

    # copy out flags, if we raised any
    stats.chunk.num_sol_flagged = n_gflags
    if n_gflags:
        # also for up message with flagging stats
        fstats = ""
        for flagname, mask in FL.categories().iteritems():
            if mask != FL.MISSING:
                n_flag = (gm.gflags&mask != 0).sum()
                if n_flag:
                    fstats += "{}:{}({:.2%}) ".format(flagname, n_flag, n_flag/float(gm.gflags.size))
        print>> log, ModColor.Str("{} solver flags raised: {}".format(label, fstats))

    return gm, (resid_arr if compute_residuals else None), stats



def solve_only(obser_arr, model_arr, flags_arr, weight_arr, chunk_ts, chunk_fs, tile, key, label, options):
    # apply weights
    if weight_arr is not None:
        obser_arr *= weight_arr[..., np.newaxis, np.newaxis]
        model_arr *= weight_arr[np.newaxis, ..., np.newaxis, np.newaxis]

    gm, _, stats = _solve_gains(obser_arr, model_arr, flags_arr, chunk_ts, chunk_fs, options, label=label)

    return gm, None, stats


def solve_and_correct(obser_arr, model_arr, flags_arr, weight_arr, chunk_ts, chunk_fs, tile, key, label, options):
    # if weights are set, multiply data and model by weights, but keep an unweighted copy
    # of the data, since we need to correct
    if weight_arr is not None:
        obser_arr1 = obser_arr*weight_arr[..., np.newaxis, np.newaxis]
        model_arr *= weight_arr[np.newaxis, ..., np.newaxis, np.newaxis]
    else:
        obser_arr1 = obser_arr

    gm, _, stats = _solve_gains(obser_arr1, model_arr, flags_arr, chunk_ts, chunk_fs, options, label=label)

    # for corrected visibilities, take the first data/model pair only
    corr_vis = np.zeros_like(obser_arr[0,...])
    gm.apply_inv_gains(obser_arr[0,...], corr_vis)

    return gm, corr_vis, stats


def solve_and_correct_res(obser_arr, model_arr, flags_arr, weight_arr, chunk_ts, chunk_fs, tile, key, label, options):
    # if weights are set, multiply data and model by weights, but keep an unweighted copy
    # of the model and data, since we need to correct the residuals

    if weight_arr is not None:
        obser_arr1 = obser_arr * weight_arr[..., np.newaxis, np.newaxis]
        model_arr1 = model_arr * weight_arr[np.newaxis, ..., np.newaxis, np.newaxis]
    else:
        obser_arr1, model_arr1 = obser_arr, model_arr

    # use the residuals computed in solve_gains() only if no weights. Otherwise need
    # to recompute them from unweighted versions
    gm, resid_vis, stats = _solve_gains(obser_arr1, model_arr1, flags_arr, chunk_ts, chunk_fs, options, label=label,
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

        chunk_ts, chunk_fs = tile.get_chunk_tfs(chunk_key)

        gm, corr_vis, stats = solver(obser_arr, model_arr, flags_arr, weight_arr, chunk_ts, chunk_fs, tile, chunk_key, label, options)

        # copy results back into tile

        tile.set_chunk_cubes(corr_vis, flags_arr if stats.chunk.num_sol_flagged else None, chunk_key)

        tile.set_chunk_gains(gm.gains, chunk_key)

        return stats
    except Exception, exc:
        print>>log,ModColor.Str("Solver for tile {} chunk {} failed with exception: {}".format(itile, label, exc))
        print>>log,traceback.format_exc()
        raise

