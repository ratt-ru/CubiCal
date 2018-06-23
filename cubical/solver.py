# CubiCal: a radio interferometric calibration suite
# (c) 2017 Rhodes University & Jonathan S. Kenyon
# http://github.com/ratt-ru/CubiCal
# This code is distributed under the terms of GPLv2, see LICENSE.md for details
"""
Implements the solver loop.
"""
import numpy as np
import traceback
from cubical.tools import logger, ModColor
from cubical.flagging import FL
from cubical.statistics import SolverStats
from cubical.tools import BREAK  # useful: can set static breakpoints by putting BREAK() in the code

log = logger.getLogger("solver")
#log.verbosity(2)

# global defaults dict
GD = None

# MS metadata
metadata = None

# gain machine factory to use
gm_factory = None

# IFR-based gain machine to use
ifrgain_machine = None

# Conversion factor for sigma = SIGMA_MAD*mad
SIGMA_MAD = 1.4826

import __builtin__
try:
    __builtin__.profile
except AttributeError:
    # No line profiler, provide a pass-through version
    def profile(func): return func
    __builtin__.profile = profile#if 'profile' not in globals():

@profile
def _solve_gains(gm, obser_arr, model_arr, flags_arr, sol_opts, label="", compute_residuals=None):
    """
    Main body of the GN/LM method. Handles iterations and convergence tests.

    Args:
        gm (:obj:`~cubical.machines.abstract_machine.MasterMachine`): 
            The gain machine which will be used in the solver loop.
        obser_arr (np.ndarray): 
            Shape (n_mod, n_tim, n_fre, n_ant, n_ant, n_cor, n_cor) array containing observed 
            visibilities. 
        model_arr (np.ndarray): 
            Shape (n_dir, n_mod, n_tim, n_fre, n_ant, n_ant, n_cor, n_cor) array containing model 
            visibilities. 
        flags_arr (np.ndarray): 
            Shape (n_tim, n_fre, n_ant, n_ant) integer array containing flag data.
        sol_opts (dict): 
            Solver options (see [sol] section in DefaultParset.cfg).
        label (str, optional):             
            Label identifying the current chunk (e.g. "D0T1F2").
        compute_residuals (bool, optional): 
            If set, the final residuals will be computed and returned.

    Returns:
        2-element tuple
            
            - resid (np.ndarray)
                The final residuals (if compute_residuals is set), else None.
            - stats (:obj:`~cubical.statistics.SolverStats`)
                An object containing solver statistics.
    """
    min_delta_g  = sol_opts["delta-g"]
    chi_tol      = sol_opts["delta-chi"]
    chi_interval = sol_opts["chi-int"]
    stall_quorum = sol_opts["stall-quorum"]


    # collect flagging options

    flag_warning_threshold = GD['flags']["warn-thr"]
    mad_flag = GD['flags']["mad-flag"]
    mad_threshold = GD['flags']["mad-thr"]
    medmad_threshold = GD['flags']["mad-med-thr"]
    if not isinstance(mad_threshold, list):
        mad_threshold = [mad_threshold]
    if not isinstance(medmad_threshold, list):
        medmad_threshold = [medmad_threshold]
    mad_diag = GD['flags']['mad-diag']
    mad_offdiag = metadata.num_corrs == 4 and GD['flags']['mad-offdiag']
    if not mad_diag and not mad_offdiag:
        mad_flag = False

    # setup MAD estimation settings
    mad_per_corr = False
    if GD['flags']['mad-estimate'] == 'corr':
        mad_per_corr = True
        mad_estimate_diag, mad_estimate_offdiag = mad_diag, mad_offdiag
    elif GD['flags']['mad-estimate'] == 'all':
        mad_estimate_diag = True
        mad_estimate_offdiag = metadata.num_corrs == 4
    elif GD['flags']['mad-estimate'] == 'diag':
        mad_estimate_diag, mad_estimate_offdiag = True, False
    elif GD['flags']['mad-estimate'] == 'offdiag':
        if metadata.num_corrs == 4:
            mad_estimate_diag, mad_estimate_offdiag = False, True
        else:
            mad_estimate_diag, mad_estimate_offdiag = True, False
    else:
        raise RuntimeError("invalid --flags-mad-estimate {} setting".format(GD['flags']['mad-estimate']))

    def get_mad_thresholds():
        """MAD thresholds above are either a list, or empty. Each time we access the list, we pop the first element,
        until the list is down to one element."""
        if not mad_flag:
            return 0, 0
        return mad_threshold.pop(0) if len(mad_threshold)>1 else (mad_threshold[0] if mad_threshold else 0), \
               medmad_threshold.pop(0) if len(medmad_threshold)>1 else (medmad_threshold[0] if medmad_threshold else 0)

    # Initialise stat object.

    stats = SolverStats(obser_arr)
    stats.chunk.label = label

    n_stall = 0
    frac_stall = 0
    n_original_flags = (flags_arr&~(FL.PRIOR|FL.MISSING) != 0).sum()

    # initialize iteration counter

    num_iter = 0

    # Estimates the overall noise level and the inverse variance per channel and per antenna as
    # noise varies across the band. This is used to normalize chi^2.

    stats.chunk.init_noise, inv_var_antchan, inv_var_ant, inv_var_chan = \
                                                        stats.estimate_noise(obser_arr, flags_arr)

    # if we have directions in the model, but the gain machine is non-DD, collapse them
    if not gm.dd_term and model_arr.shape[0] > 1:
        model_arr = model_arr.sum(axis=0, keepdims=True)

    # This works out the conditioning of the solution, sets up various chi-sq normalization
    # factors etc, and does any other precomputation required by the current gain machine.

    gm.precompute_attributes(model_arr, flags_arr, inv_var_chan)

    def get_flagging_stats():
        """Returns a string describing per-flagset statistics"""
        fstats = []

        for flag, mask in FL.categories().iteritems():
            n_flag = ((flags_arr & mask) != 0).sum()
            if n_flag:
                fstats.append("{}:{}({:.2%})".format(flag, n_flag, n_flag/float(flags_arr.size)))

        return " ".join(fstats)

    def update_stats(flags, statfields):
        """
        This function updates the solver stats object with a count of valid data points used for chi-sq 
        calculations
        """
        unflagged = (flags==0)
        # Compute number of terms in each chi-square sum. Shape is (n_tim, n_fre, n_ant).

        nterms  = 2 * gm.n_cor * gm.n_cor * np.sum(unflagged, axis=3)

        # Update stats object accordingly.

        for field in statfields:
            getattr(stats.chanant,  field+'n')[...] = np.sum(nterms, axis=0)
            getattr(stats.timeant,  field+'n')[...] = np.sum(nterms, axis=1)
            getattr(stats.timechan, field+'n')[...] = np.sum(nterms, axis=2)
    
    update_stats(flags_arr, ('initchi2', 'chi2'))

    # In the event that there are no solutions with valid data, this will log some of the
    # flag information and break out of the function.

    if not gm.has_valid_solutions:
        stats.chunk.num_sol_flagged, _ = gm.num_gain_flags()

        print>> log, ModColor.Str("{} no solutions: {}; flags {}".format(label,
                        gm.conditioning_status_string, get_flagging_stats()))
        return (obser_arr if compute_residuals else None), stats

    # Initialize a residual array.

    resid_shape = [gm.n_mod, gm.n_tim, gm.n_fre, gm.n_ant, gm.n_ant, gm.n_cor, gm.n_cor]

    resid_arr = gm.cykernel.allocate_vis_array(resid_shape, obser_arr.dtype, zeros=True)
    gm.compute_residual(obser_arr, model_arr, resid_arr)
    resid_arr[:,flags_arr!=0] = 0

    # This flag is set to True when we have an up-to-date residual in resid_arr.
    
    have_residuals = True

    stats.chunk.num_mad_flagged = 0

    # apply MAD flagging
    global _madmax_plotnum
    _madmax_plotnum = 0

    @profile
    def beyond_thunderdome(max_label, threshold, med_threshold):
        """This function implements MAD-based flagging on residuals"""
        import cubical.kernels
        cymadmax = cubical.kernels.import_kernel("cymadmax")
        # estimate MAD of off-diagonal elements
        absres = np.empty_like(resid_arr, dtype=np.float32)
        np.abs(resid_arr, out=absres)
        if mad_per_corr:
            mad, goodies = cymadmax.compute_mad_per_corr(absres, flags_arr, diag=mad_estimate_diag, offdiag=mad_estimate_offdiag)
        else:
            mad, goodies = cymadmax.compute_mad(absres, flags_arr, diag=mad_estimate_diag, offdiag=mad_estimate_offdiag)
        # any of it non-zero?
        if mad.mask.all():
            return
        # estimate median MAD
        medmad = np.ma.median(mad, axis=(1,2))
        # all this was worth it, just so I could type "mad.max()" as legit code
        print>>log(2),"{} per-baseline MAD min {:.2f}, max {:.2f}, median {:.2f}".format(max_label, mad.min(), mad.max(), np.ma.median(medmad))
        if log.verbosity() > 4:
            for imod in xrange(gm.n_mod):
                if mad_per_corr:
                    for ic1,c1 in enumerate(metadata.feeds):
                        for ic2,c2 in enumerate(metadata.feeds):
                            per_bl = [(mad[imod,p,q,ic1,ic2], p, q) for p in xrange(gm.n_ant)
                                      for q in xrange(p+1, gm.n_ant) if not mad.mask[imod,p,q,ic1,ic2]]
                            per_bl = ["{} ({}m): {:.2f}".format(metadata.baseline_name[p,q], int(metadata.baseline_length[p,q]), x)
                                      for x, p, q in sorted(per_bl)[::-1]]
                            print>>log(4),"{} model {} {}{} MADs are {}".format(label, imod,
                                                                                c1.upper(), c2.upper(), ", ".join(per_bl))
                else:
                    per_bl = [(mad[imod,p,q,], p, q) for p in xrange(gm.n_ant)
                              for q in xrange(p+1, gm.n_ant) if not mad.mask[imod,p,q]]
                    per_bl = ["{} ({}m) {:.2f}".format(metadata.baseline_name[p,q], int(metadata.baseline_length[p,q]), x)
                              for x, p, q in sorted(per_bl)[::-1]]
                    print>>log(4),"{} model {} MADs are {}".format(label, imod, ", ".join(per_bl))

        @profile
        def kill_the_bad_guys(baddies, method):
            nbad = int(baddies.sum())
            stats.chunk.num_mad_flagged += nbad
            if nbad:
                if nbad < flags_arr.size * flag_warning_threshold:
                    warning, color = "", "blue"
                else:
                    warning, color = "WARNING: ", "red"
                print>> log(1, color), "{}{} {} kills {} ({:.2%}) visibilities".format(warning, max_label, method, nbad,
                                        nbad/float(baddies.size))
                if log.verbosity() > 2 or GD['flags']['mad-plot']:
                    per_bl = []
                    total_elements = float(gm.n_tim * gm.n_fre)
                    for p in xrange(gm.n_ant):
                        for q in xrange(p + 1, gm.n_ant):
                            n_flagged = baddies[:, :, p, q].sum()
                            if n_flagged:
                                per_bl.append((n_flagged, p ,q))
                    per_bl = sorted(per_bl, reverse=True)
                    # print
                    per_bl_str = ["{} ({}m): {} ({:.2%})".format(metadata.baseline_name[p,q],
                                    int(metadata.baseline_length[p,q]), n_flagged, n_flagged/total_elements)
                                  for n_flagged, p, q in per_bl]
                    print>> log(3), "{} of which per baseline: {}".format(label, ", ".join(per_bl_str))
                    # plot, if asked to
                    if GD['flags']['mad-plot']:
                        if len(per_bl) < 3:
                            baselines_to_plot = [ (0, "worst") ]
                        else:
                            baselines_to_plot = [ (0, "worst"), (len(per_bl)//2, "median") ]
                        import pylab
                        for ibl, baseline_label in baselines_to_plot:
                            n_flagged, p, q = per_bl[ibl]
                            fraction = n_flagged / total_elements
                            if fraction <= GD['flags']['mad-plot-thr']:
                                continue
                            blname = metadata.baseline_name[p,q]
                            bllen  = int(metadata.baseline_length[p,q])
                            feeds =  metadata.feeds
                            # inv: data that was flagged prior to this mad max step
                            fl_prior = (flags_arr[:,:,p,q]!=0)&~baddies[:,:,p,q]
                            pylab.figure(figsize=(16,10))
                            resmask = np.zeros_like(absres[0, :, :, p, q], dtype=bool)
                            resmask[:] = fl_prior[...,np.newaxis,np.newaxis]
                            res = np.ma.masked_array(absres[0, :, :, p, q], resmask)
                            vmin = res.min()
                            vmax = res.max()
                            from matplotlib.colors import LogNorm
                            norm = LogNorm(vmin, vmax)
                            for c1,x1 in enumerate(feeds.upper()):
                                for c2,x2 in enumerate(feeds.upper()):
                                    pylab.subplot(2, 4, 1+c1*2+c2)
                                    pylab.imshow(res[...,c1,c2], norm=norm, aspect='auto')
                                    mm = mad[0,p,q,c1,c2] if mad_per_corr else mad[0,p,q]
                                    pylab.title("{}{} residuals (MAD {:.2f})".format(x1, x2, mm))
                                    pylab.colorbar()
                            for c1,x1 in enumerate(feeds.upper()):
                                for c2,x2 in enumerate(feeds.upper()):
                                    pylab.subplot(2, 4, 5+c1*2+c2)
                                    pylab.imshow(np.ma.masked_array(absres[0, :, :, p, q, c1, c2], fl_prior|baddies[:, :, p, q]),
                                                 norm=norm, aspect='auto')
                                    pylab.title("{}{} flagged".format(x1, x2))
                                    pylab.colorbar()
                            pylab.suptitle("{} {}: baseline {} ({}m), {} ({:.2%}) visibilities killed ({} case)".format(max_label,
                                            method, blname, bllen, n_flagged, fraction, baseline_label))
                            if GD['flags']['mad-plot'] == 'show':
                                pylab.show()
                            else:
                                global _madmax_plotnum
                                filename = '{}.{}.madmax.{}.png'.format(GD['out']['name'], label, _madmax_plotnum)
                                pylab.savefig(filename, dpi=300)
                                _madmax_plotnum += 1
                                print>>log(1),"{}: saving Mad Max flagging plot to {}".format(label,filename)
            else:
                print>> log(2),"{} {} abides".format(max_label, method)

        thr = np.zeros((gm.n_mod, gm.n_ant, gm.n_ant, gm.n_cor, gm.n_cor), dtype=np.float32)
        # apply per-baseline MAD threshold
        if threshold:
            if mad_per_corr:
                thr[:] = threshold * mad / SIGMA_MAD
            else:
                thr[:] = threshold * mad[...,np.newaxis,np.newaxis] / SIGMA_MAD
            baddies = cymadmax.threshold_mad(absres, thr, flags_arr, FL.MAD, goodies, diag=mad_diag, offdiag=mad_offdiag)
            kill_the_bad_guys(baddies, "baseline-based Mad Max ({} sigma)".format(threshold))

        # apply global median MAD threshold
        if med_threshold:
            if mad_per_corr:
                thr[:] = med_threshold * medmad[:,np.newaxis,np.newaxis,:,:] / SIGMA_MAD
            else:
                thr[:] = med_threshold * medmad[:,np.newaxis,np.newaxis,np.newaxis,np.newaxis] / SIGMA_MAD
            baddies = cymadmax.threshold_mad(absres, thr, flags_arr, FL.MAD, goodies, diag=mad_diag, offdiag=mad_offdiag)
            kill_the_bad_guys(baddies, "global Mad Max ({} sigma)".format(med_threshold))

    # do mad max flagging, if requested
    thr1, thr2 = get_mad_thresholds()
    if thr1 or thr2:
        beyond_thunderdome("{} initial".format(label), thr1, thr2)

    def compute_chisq(statfield=None):
        """
        Computes chi-squared statistic based on current residuals and noise estimates.
        Populates the stats object with it.
        """
        chisq, chisq_per_tf_slot, chisq_tot = gm.compute_chisq(resid_arr, inv_var_chan)

        if statfield:
            getattr(stats.chanant, statfield)[...]  = np.sum(chisq, axis=0)
            getattr(stats.timeant, statfield)[...]  = np.sum(chisq, axis=1)
            getattr(stats.timechan, statfield)[...] = np.sum(chisq, axis=2)
        
        return chisq_per_tf_slot, chisq_tot

    chi, mean_chi = compute_chisq(statfield='initchi2')
    stats.chunk.init_chi2 = mean_chi

    # The following provides conditioning information when verbose is set to > 0.
    if log.verbosity() > 0:

        print>> log, "{} chi^2_0 {:.4}; {}; noise {:.3}, flags: {}".format(
                        label, mean_chi, gm.conditioning_status_string,
                        float(stats.chunk.init_noise), get_flagging_stats())

    # Main loop of the NNLS method. Terminates after quorum is reached in either converged or
    # stalled solutions or when the maximum number of iterations is exceeded.

    while not(gm.has_converged) and not(gm.has_stalled):

        num_iter = gm.next_iteration()

        # This is currently an awkward necessity - if we have a chain of jones terms, we need to 
        # make sure that the active term is correct and need to support some sort of decision making
        # for testing convergence. I think doing the iter increment here might be the best choice,
        # with an additional bit of functionality for Jones chains. I suspect I will still need to 
        # change the while loop component to be compatible with the idea of partial convergence.
        # Perhaps this should all be done right at the top of the function? A better idea is to let
        # individual machines be aware of their own stalled/converged status, and make those
        # properties more complicated on the chain. This should allow for fairly easy substitution 
        # between the various machines.

        gm.compute_update(model_arr, obser_arr)
        
        # flag solutions. This returns True if any flags have been propagated out to the data.
        if gm.flag_solutions(flags_arr, False):

            update_stats(flags_arr, ('chi2',))

            # Re-zero the model and data at newly flagged points.
            # TODO: is this needed?
            # TODO: should we perhaps just zero the model per flagged direction, and only flag the data?
            # OMS: probably not: flag propagation is now handled inside the gain machine. If a flag is
            # propagated out to the data, then that slot is gone gone gone and should be zeroe'd everywhere.
            
            new_flags = flags_arr&~(FL.MISSING|FL.PRIOR) !=0
            model_arr[:, :, new_flags, :, :] = 0
            obser_arr[   :, new_flags, :, :] = 0

            # Break out of the solver loop if we find ourselves with no valid solution intervals.
            
            if not gm.has_valid_solutions:
                break

        # print>>log,"{} {} {}".format(de.gains[1,5,2,5], de.posterior_gain_error[1,5,2,5], de.posterior_gain_error[1].mean())
        #
        have_residuals = False

        # Compute values used in convergence tests. This check implicitly marks flagged gains as 
        # converged.
        
        gm.check_convergence(min_delta_g)

        # Check residual behaviour after a number of iterations equal to chi_interval. This is
        # expensive, so we do it as infrequently as possible.

        if (num_iter % chi_interval) == 0:

            old_chi, old_mean_chi = chi, mean_chi

            gm.compute_residual(obser_arr, model_arr, resid_arr)
            resid_arr[:,flags_arr!=0] = 0

            # do mad max flagging, if requested
            thr1, thr2 = get_mad_thresholds()
            if thr1 or thr2:
                beyond_thunderdome("{} iter {}".format(label, num_iter), thr1, thr2)

            chi, mean_chi = compute_chisq()

            have_residuals = True

            # Check for stalled solutions - solutions for which the residual is no longer improving.

            n_stall = float(np.sum(((old_chi - chi) < chi_tol*old_chi)))
            frac_stall = n_stall/chi.size

            gm.has_stalled = (frac_stall >= stall_quorum)

            if log.verbosity() > 1:

                delta_chi = (old_mean_chi-mean_chi)/old_mean_chi

                print>> log(2), ("{} {} chi2 {:.4}, delta {:.4}, stall {:.2%}").format(
                                    label, gm.current_convergence_status_string,
                                    mean_chi, delta_chi, frac_stall)

    # num_valid_solutions will go to 0 if all solution intervals were flagged. If this is not the
    # case, generate residuals etc.

    if gm.has_valid_solutions:
        # Final round of flagging
        flagged = gm.flag_solutions(flags_arr, True)
    else:
        flagged = None
        
    # check this again, because final round of flagging could have killed us
    if gm.has_valid_solutions:
        # Do we need to recompute the final residuals?
        if (sol_opts['last-rites'] or compute_residuals) and (not have_residuals or flagged):
            gm.compute_residual(obser_arr, model_arr, resid_arr)
            resid_arr[:,flags_arr!=0] = 0

            # do mad max flagging, if requested
            thr1, thr2 = get_mad_thresholds()
            if thr1 or thr2:
                beyond_thunderdome("{} final".format(label), thr1, thr2)

            if sol_opts['last-rites']:
                # Recompute chi-squared based on original noise statistics.
                chi, mean_chi = compute_chisq(statfield='chi2')

        # Re-estimate the noise using the final residuals, if last rites are needed.

        if sol_opts['last-rites']:
            stats.chunk.noise, inv_var_antchan, inv_var_ant, inv_var_chan = \
                                        stats.estimate_noise(resid_arr, flags_arr, residuals=True)
            chi1, mean_chi1 = compute_chisq(statfield='chi2')

        stats.chunk.chi2 = mean_chi

        message = "{} {}, stall {:.2%}, chi^2 {:.4} -> {:.4}".format(label,
                    gm.final_convergence_status_string,
                    frac_stall, float(stats.chunk.init_chi2), mean_chi)


        if sol_opts['last-rites']:

            message = "{} ({:.4}), noise {:.3} -> {:.3}".format(message,
                            float(mean_chi1), float(stats.chunk.init_noise), float(stats.chunk.noise))

        print>> log, message

    # If everything has been flagged, no valid solutions are generated. 

    else:
        
        print>>log(0, "red"), "{} {}: completely flagged?".format(label, gm.final_convergence_status_string)

        stats.chunk.chi2 = 0
        resid_arr = obser_arr

    stats.chunk.iters = num_iter
    stats.chunk.num_converged = gm.num_converged_solutions
    stats.chunk.num_stalled = n_stall

    # collect messages from various flagging sources, and print to log if any
    flagstatus = []

    stats.chunk.num_sol_flagged, _ = gm.num_gain_flags()
    if stats.chunk.num_sol_flagged:
        # also for up message with flagging stats
        fstats = ""
        for flagname, mask in FL.categories().iteritems():
            if mask != FL.MISSING:
                n_flag, n_tot = gm.num_gain_flags(mask)
                if n_flag:
                    fstats += "{}:{}({:.2%}) ".format(flagname, n_flag, n_flag/float(n_tot))

        flagstatus.append("solver flags {}".format(fstats))

    if stats.chunk.num_mad_flagged:
        flagstatus.append("Mad Max took out {} visibilities".format(stats.chunk.num_mad_flagged))

    if flagstatus:
        n_new_flags = (flags_arr&~(FL.PRIOR | FL.MISSING) != 0).sum() - n_original_flags
        if n_new_flags < flags_arr.size*flag_warning_threshold:
            warning, color = "", "blue"
        else:
            warning, color = "WARNING: ", "red"
        print>> log(0, color), "{}{} {}: {} ({:.2%}) new data flags".format(
            warning, label, ", ".join(flagstatus),
            n_new_flags, n_new_flags / float(flags_arr.size))


    return (resid_arr if compute_residuals else None), stats


class _VisDataManager(object):
    """A _VisDataManager object holds data, model, flag and weight arrays associated with a single
    chunk of visibility data. It also holds a GainMachine. It provides methods and properties for 
    computing/caching various derived arrays (weighted versions of data and model, corrupt models, etc.) 
    on demand.
    
    _VisDataManagers are used to unify the interface to the various solving methods defined below. 
    """
    def __init__(self, obser_arr, model_arr, flags_arr, weight_arr, freq_slice):
        """
        Initialises a VisDataManager.

        Args:
            obser_arr (np.ndarray): 
                Shape (n_tim, n_fre, n_ant, n_ant, n_cor, n_cor) complex array containing observed visibilities. 
            model_arr (np.ndarray): 
                Shape (n_dir, n_mod, n_tim, n_fre, n_ant, n_ant, n_cor, n_cor) complex array containing model 
                visibilities. 
            flags_arr (np.ndarray): 
                Shape (n_tim, n_fre, n_ant, n_ant) integer array containing flags.
            weight_arr (np.ndarray): 
                Shape (n_mod, n_tim, n_fre, n_ant, n_ant, n_cor, n_cor) float array containing weights.
            freq_slice (slice): 
                Slice into the full data frequency axis corresponding to this chunk. 
        """
        self.gm = None
        self.obser_arr, self.model_arr, self.flags_arr, self.weight_arr = \
            obser_arr, model_arr, flags_arr, weight_arr
        self._wobs_arr = self._wmod_arr = None
        self.freq_slice = freq_slice
        self._model_corrupted = False

    @property
    def weighted_obser(self):
        """
        This property gives the observed visibilities times the weights
        
        Returns:
            Weighted observed visibilities (np.ndarray) of shape (n_mod, n_tim, n_fre, n_ant, n_ant, n_cor, n_cor)
        """
        if self._wobs_arr is None:
            if self.weight_arr is not None:
                self._wobs_arr = self.obser_arr[np.newaxis,...] * self.weight_arr[..., np.newaxis, np.newaxis]
            else:
                self._wobs_arr = self.obser_arr.copy().reshape([1]+list(self.obser_arr.shape))
                # zero the flagged visibilities. Note that if we have a weight, this is not necessary,
                # as they will already get zero weight in data_handler
                self._wobs_arr[:, self.flags_arr!=0, :, :] = 0
        return self._wobs_arr

    @property
    def weighted_model(self):
        """
        This property gives the model visibilities times the weights

        Returns:
            Weighted model visibilities (np.ndarray) of shape (n_dir, n_mod, n_tim, n_fre, n_ant, n_ant, n_cor, n_cor)
        """
        if self._wmod_arr is None:
            if self.weight_arr is not None:
                self._wmod_arr = self.model_arr * self.weight_arr[np.newaxis, ..., np.newaxis, np.newaxis]
            else:
                self._wmod_arr = self.model_arr.copy()
                # zero the flagged visibilities. Note that if we have a weight, this is not necessary,
                # as they will already get zero weight in data_handler
                self._wmod_arr[:, :, self.flags_arr!=0, :, :] = 0
        return self._wmod_arr

    @property
    def corrupt_weighted_model(self):
        """
        This property gives the model visibilities, corrupted by the gains, times the weights, summed over directions.

        Returns:
            Weighted corrupted model visibilities (np.ndarray) of shape (n_mod, n_tim, n_fre, n_ant, n_ant, n_cor, n_cor)
        """
        cmod = self.corrupt_model(None).sum(0)
        if self.weight_arr is not None:
            return cmod*self.weight_arr[..., np.newaxis, np.newaxis]
        else:
            return cmod

    def corrupt_residual(self, imod=0, idir=slice(None)):
        """
        This method returns the (corrupted) residual with respect to a given model
        
        Args:
            imod (int): 
                Index of model (0 to n_mod-1). 

        Returns:
            Weighted residual visibilities (np.ndarray) of shape (n_tim, n_fre, n_ant, n_ant, n_cor, n_cor)
        """
        return self.obser_arr - self.corrupt_model(imod, idir)


    def corrupt_model(self, imod=0, idir=slice(None)):
        """
        This method returns the model visibilities, corrupted by the gains. 
        The first time it is called, the model is corrupted in-place. 
        Args:
            imod (int or None): 
                Index of model (0 to n_mod-1), or None to return all models
            idir (slice or list)
                Directions to include in corrupted moel

        Returns:
            Corrupted model visibilities (np.ndarray) of shape (n_mod, n_tim, n_fre, n_ant, n_ant, n_cor, n_cor)
            if imod is None, otherwise the model axis is omitted.
        """
        if not self._model_corrupted:
            self.gm.apply_gains(self.model_arr)
            self._model_corrupted = True
        if imod is None:
            return self.model_arr
        return self.model_arr[idir, imod].sum(0)


def solve_only(vdm, soldict, label, sol_opts):
    """
    Run the solver and neither save nor apply solutions. 

    Args:
        vdm (:obj:`_VisDataManager`): 
            VisDataManager for this chunk of data
        soldict (:obj:~cubical.tools.shared_dict.SharedDict): 
            Shared dict used to pass solutions (IFR gain solutions, primarily) back out to the
            calling thread. 
        label (str):             
            Label identifying the current chunk (e.g. "D0T1F2").
        sol_opts (dict): 
            Solver options (see [sol] section in DefaultParset.cfg).        

    Returns:
        2-element tuple

            - _ (None) 
                None (required for compatibility)
            - stats (:obj:`~cubical.statistics.SolverStats`)
                An object containing solver statistics.
    """

    _, stats = _solve_gains(vdm.gm, vdm.weighted_obser, vdm.weighted_model, vdm.flags_arr, sol_opts, label=label)
    if ifrgain_machine.is_computing():
        ifrgain_machine.update(vdm.weighted_obser, vdm.corrupt_weighted_model, vdm.flags_arr, vdm.freq_slice, soldict)

    return None, stats


def solve_and_correct(vdm, soldict, label, sol_opts):
    """
    Run the solver and save and apply the resulting gain solutions to the observed data. Produces
    corrected data. 
    
    Args:
        vdm (:obj:`_VisDataManager`): 
            VisDataManager for this chunk of data
        soldict (:obj:~cubical.tools.shared_dict.SharedDict): 
            Shared dict used to pass solutions (IFR gain solutions, primarily) back out to the
            calling thread. 
        label (str):             
            Label identifying the current chunk (e.g. "D0T1F2").
        sol_opts (dict): 
            Solver options (see [sol] section in DefaultParset.cfg).        
    
    Returns:
        2-element tuple
            
            - corr_vis (np.ndarray) 
                Shape (n_tim, n_fre, n_ant, n_ant, n_cor, n_cor) array containing corrected 
                visibilities. 
            - stats (:obj:`~cubical.statistics.SolverStats`)
                An object containing solver statistics.
    """

    _, stats = _solve_gains(vdm.gm, vdm.weighted_obser, vdm.weighted_model, vdm.flags_arr, sol_opts, label=label)

    # for corrected visibilities, take the first data/model pair only
    corr_vis = np.zeros_like(vdm.obser_arr)
    vdm.gm.apply_inv_gains(vdm.obser_arr, corr_vis)

    if ifrgain_machine.is_computing():
        ifrgain_machine.update(vdm.weighted_obser, vdm.corrupt_weighted_model, vdm.flags_arr, vdm.freq_slice, soldict)

    return corr_vis, stats


def solve_and_correct_residuals(vdm, soldict, label, sol_opts, correct=True):
    """
    Run the solver, generate residuals, and (optionally) apply the resulting gain solutions to the residuals. 
    Produces (un)corrected residuals. 

    Args:
        vdm (:obj:`_VisDataManager`): 
            VisDataManager for this chunk of data
        soldict (:obj:~cubical.tools.shared_dict.SharedDict): 
            Shared dict used to pass solutions (IFR gain solutions, primarily) back out to the
            calling thread. 
        label (str):             
            Label identifying the current chunk (e.g. "D0T1F2").
        sol_opts (dict): 
            Solver options (see [sol] section in DefaultParset.cfg).        
        correct (bool):
            If True, residuals are corrected

    Returns:
        2-element tuple
            
            - res_vis (np.ndarray)
                Shape (n_tim, n_fre, n_ant, n_ant, n_cor, n_cor) array containing (un)corrected 
                residuals. 
            - stats (:obj:`~cubical.statistics.SolverStats`)
                An object containing solver statistics.
    """

    # use the residuals computed in solve_gains() only if no weights. Otherwise need
    # to recompute them from unweighted versions
    resid_vis, stats = _solve_gains(vdm.gm, vdm.weighted_obser, vdm.weighted_model, vdm.flags_arr,
                                        sol_opts, label=label, compute_residuals=False)

    # compute IFR gains, if needed. Note that this computes corrupt models, so it makes sense
    # doing it before recomputing the residuals: saves time
    if ifrgain_machine.is_computing():
        ifrgain_machine.update(vdm.weighted_obser, vdm.corrupt_weighted_model, vdm.flags_arr, vdm.freq_slice, soldict)

    # compute residuals
    resid_vis = vdm.corrupt_residual(sol_opts["subtract-model"],  sol_opts["subtract-dirs"])

    # correct residual if required
    if correct:
        corr_vis = np.zeros_like(resid_vis)
        vdm.gm.apply_inv_gains(resid_vis, corr_vis)
        return corr_vis, stats
    else:
        return resid_vis, stats

def solve_and_subtract(*args, **kw):
    """
    Run the solver, generate residuals. Produces uncorrected residuals. Equivalent to calling
    solve_and_correct_residuals(..., correct=False)
    """
    return solve_and_correct_residuals(correct=False, *args, **kw)


def correct_only(vdm, soldict, label, sol_opts):
    """
    Do not solve. Apply priot gain solutions to the observed data, generating corrected data.
    
    Args:
        vdm (:obj:`_VisDataManager`): 
            VisDataManager for this chunk of data
        soldict (:obj:~cubical.tools.shared_dict.SharedDict): 
            Shared dict used to pass solutions (IFR gain solutions, primarily) back out to the
            calling thread. 
        label (str):             
            Label identifying the current chunk (e.g. "D0T1F2").
        sol_opts (dict): 
            Solver options (see [sol] section in DefaultParset.cfg).        
            
    Returns:
        2-element tuple
            
            - corr_vis (np.ndarray)
                Shape (n_tim, n_fre, n_ant, n_ant, n_cor, n_cor) array containing corrected visibilities. 
            - _ (None)
                None (required for compatibility)
    """

    corr_vis = np.zeros_like(vdm.obser_arr)
    vdm.gm.apply_inv_gains(vdm.obser_arr, corr_vis)

    if vdm.model_arr is not None and ifrgain_machine.is_computing():
        ifrgain_machine.update(vdm.weighted_obser, vdm.corrupt_weighted_model, vdm.flags_arr, vdm.freq_slice, soldict)

    return corr_vis, None


def correct_residuals(vdm, soldict, label, sol_opts, correct=True):
    """
    Do not solve. Apply prior gain solutions, generate (un)corrected residuals.

    Args:
        vdm (:obj:`_VisDataManager`): 
            VisDataManager for this chunk of data
        soldict (:obj:~cubical.tools.shared_dict.SharedDict): 
            Shared dict used to pass solutions (IFR gain solutions, primarily) back out to the
            calling thread. 
        label (str):             
            Label identifying the current chunk (e.g. "D0T1F2").
        sol_opts (dict): 
            Solver options (see [sol] section in DefaultParset.cfg).        
        correct (bool):
            If True, residuals are corrected

    Returns:
        2-element tuple

            - resid_vis (np.ndarray)
                Shape (n_tim, n_fre, n_ant, n_ant, n_cor, n_cor) array containing (un)corrected residuals. 
            - _ (None)
                None (required for compatibility)
    """
    # compute IFR gains, if needed. Note that this computes corrupt models, so it makes sense
    # doing it before recomputing the residuals: saves time
    if ifrgain_machine.is_computing():
        ifrgain_machine.update(vdm.weighted_obser, vdm.corrupt_weighted_model, vdm.flags_arr, vdm.freq_slice, soldict)

    resid_vis = vdm.corrupt_residual(sol_opts["subtract-model"],  sol_opts["subtract-dirs"])

    # correct residual if required
    if correct:
        corr_vis = np.zeros_like(resid_vis)
        vdm.gm.apply_inv_gains(resid_vis, corr_vis)
        return corr_vis, None
    else:
        return resid_vis, None

def subtract_only(*args, **kw):
    """
    Do not solve. Apply prior gain solutions, generate uncorrected residuals. Equivalent to calling
    correct_residuals(..., correct=False)
    """
    return correct_residuals(correct=False, *args, **kw)


SOLVERS = { 'so': solve_only,
            'sc': solve_and_correct,
            'sr': solve_and_correct_residuals,
            'ss': solve_and_subtract,
            'ac': correct_only,
            'ar': correct_residuals,
            'as': subtract_only
            }


#@profile
def run_solver(solver_type, itile, chunk_key, sol_opts, debug_opts):
    """
    Initialises a gain machine and invokes the solver for the current chunk.

    Args:
        solver_type (str):
            Specifies type of solver to use.
        itile (int):
            Index of current Tile object.
        chunk_key (str):
            Label identifying the current chunk (e.g. "D0T1F2").
        sol_opts (dict):
            Solver options (see [sol] section in DefaultParset.cfg).

    Returns:
        :obj:`~cubical.statistics.SolverStats`:
            An object containing solver statistics.

    Raises:
        RuntimeError:
            If gain factory has not been initialised.
    """
    import cubical.workers
    cubical.workers._init_worker()

    label = None
    try:
        tile = cubical.workers.tile_list[itile]
        label = chunk_key
        solver = SOLVERS[solver_type]
        # initialize the gain machine for this chunk

        if gm_factory is None:
            raise RuntimeError("Gain machine factory has not been initialized")

        # Get chunk data from tile.

        # need to know which kernel to use to allocate visibility and flag arrays
        kernel = gm_factory.get_kernel()

        obser_arr, model_arr, flags_arr, weight_arr = tile.get_chunk_cubes(chunk_key,
                                 gm_factory.ctype,
                                 allocator=kernel.allocate_vis_array,
                                 flag_allocator=kernel.allocate_flag_array)
        
        chunk_ts, chunk_fs, _, freq_slice = tile.get_chunk_tfs(chunk_key)

        # apply IFR-based gains, if any
        ifrgain_machine.apply(obser_arr, freq_slice)

        # create subdict in shared dict for solutions etc.
        soldict = tile.create_solutions_chunk_dict(chunk_key)

        # create VisDataManager for this chunk

        vdm = _VisDataManager(obser_arr, model_arr, flags_arr, weight_arr, freq_slice)

        n_dir, n_mod = model_arr.shape[0:2] if model_arr is not None else (1,1)

        # create GainMachine
        vdm.gm = gm_factory.create_machine(vdm.weighted_obser, n_dir, n_mod, chunk_ts, chunk_fs, label)

        # Invoke solver method
        if debug_opts['stop-before-solver']:
            import pdb
            pdb.set_trace()

        corr_vis, stats = solver(vdm, soldict, label, sol_opts)
        
        # Panic if amplitude has gone crazy
        
        if debug_opts['panic-amplitude']:
            if corr_vis is not None:
                unflagged = flags_arr==0
                if unflagged.any() and abs(corr_vis[unflagged,:,:]).max() > debug_opts['panic-amplitude']:
                    raise RuntimeError("excessive amplitude in chunk {}".format(label))

        # Copy results back into tile.
        have_new_flags = stats and ( stats.chunk.num_sol_flagged > 0 or stats.chunk.num_mad_flagged > 0)

        tile.set_chunk_cubes(corr_vis, flags_arr if have_new_flags else None, chunk_key)

        # Ask the gain machine to store its solutions in the shared dict.
        gm_factory.export_solutions(vdm.gm, soldict)

        return stats

    except Exception, exc:
        print>>log,ModColor.Str("Solver for tile {} chunk {} failed with exception: {}".format(itile, label, exc))
        print>>log,traceback.format_exc()
        raise

