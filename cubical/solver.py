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
from cubical.data_handler import Tile
from cubical.flagging import FL
from cubical.machines import jones_chain_machine
from cubical.statistics import SolverStats
from pdb import set_trace as BREAK  # useful: can set static breakpoints by putting BREAK() in the code

log = logger.getLogger("solver")
#log.verbosity(2)

# gain machine factory to use
gm_factory = None

# IFR-based gain machine to use
ifrgain_machine = None


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

    # Initialise stat object.

    stats = SolverStats(obser_arr)
    stats.chunk.label = label

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

            n_flag = np.sum((flags_arr & mask) != 0)
            fstats += ("%s:%d(%.2f%%) " % (flag, n_flag, n_flag*100./flags_arr.size)) if n_flag else ""

        print>> log, ModColor.Str("{} is completely flagged: {}".format(label, fstats))

        return (obser_arr if compute_residuals else None), stats

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

        norm_factor = np.zeros_like(eqs_per_tf_slot, dtype=resid_arr.real.dtype)
        norm_factor[eqs_per_tf_slot>0] = 1./eqs_per_tf_slot[eqs_per_tf_slot>0]

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

    n_gflags = (gm.gflags&~FL.MISSING != 0).sum()

    # Do any precomputation required by the current gain machine.

    gm.precompute_attributes(model_arr)

    # Main loop of the NNLS method. Terminates after quorum is reached in either converged or
    # stalled solutions or when the maximum number of iterations is exceeded.

    while not(gm.has_converged) and not(gm.has_stalled):

        gm.update_term()

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
        
        gm.flag_solutions()

        if gm.dd_term:
            gm.gains[np.where(gm.gflags==1)] = np.eye(2)

        # In the DD case, it may be necessary to set flagged gains to zero during the loop, but
        # set all flagged terms to identity before applying them.

        # If the number of flags had increased, these need to be propagated out to the data. Note
        # that gain flags are per-direction whereas data flags are per visibility. Currently,
        # everything is flagged if any direction is flagged.

        # We remove the FL.MISSING bit when propagating as this bit is pre-set for data flagged 
        # as PRIOR|MISSING. This prevents every PRIOR but not MISSING flag from becoming MISSING.

        if gm.n_flagged > n_gflags and not(gm.dd_term):
            
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

        if (gm.iters % chi_interval) == 0:

            old_chi, old_mean_chi = chi, mean_chi

            gm.compute_residual(obser_arr, model_arr, resid_arr)

            chi, mean_chi = compute_chisq()

            have_residuals = True

            # Check for stalled solutions - solutions for which the residual is no longer improving.

            n_stall = float(np.sum(((old_chi - chi) < chi_tol*old_chi)))

            gm.has_stalled = (n_stall/n_tf_slots >= stall_quorum)

            if log.verbosity() > 1:

                delta_chi = (old_mean_chi-mean_chi)/old_mean_chi

                logvars = (label, gm.iters, mean_chi, delta_chi, gm.max_update, gm.n_cnvgd/gm.n_sols,
                           n_stall/n_tf_slots, n_gflags/float(gm.gflags.size),
                           gm.missing_gain_fraction)

                print>> log, ("{} iter {} chi2 {:.4} delta {:.4}, max gain update {:.4}, "
                              "conv {:.2%}, stall {:.2%}, g/fl {:.2%}, d/fl {:.2}%").format(*logvars)

    # num_valid_intervals will go to 0 if all solution intervals were flagged. If this is not the 
    # case, generate residuals etc.
    
    if gm.num_valid_intervals:

        # Do we need to recompute the final residuals?
        if (sol_opts['last-rites'] or compute_residuals) and not have_residuals:
            gm.compute_residual(obser_arr, model_arr, resid_arr)
            if sol_opts['last-rites']:
                # Recompute chi-squared based on original noise statistics.
                chi, mean_chi = compute_chisq(statfield='chi2')

        # Re-estimate the noise using the final residuals, if last rites are needed.

        if sol_opts['last-rites']:
            stats.chunk.noise, inv_var_antchan, inv_var_ant, inv_var_chan = \
                                        stats.estimate_noise(resid_arr, flags_arr, residuals=True)
            chi1, mean_chi1 = compute_chisq(statfield='chi2')

        stats.chunk.chi2 = mean_chi

        if isinstance(gm, jones_chain_machine.JonesChain):
            termstring = ""
            for term in gm.jones_terms:
                termstring += "{}: {} iters, conv {:.2%} ".format(term.jones_label, term.iters,
                                                                  term.n_cnvgd/term.n_sols)
        else:
            termstring = "{} iters, conv {:.2%}".format(gm.iters, gm.n_cnvgd/gm.n_sols) 

        logvars = (label, termstring, n_stall/n_tf_slots, n_gflags/float(gm.gflags.size), 
                   gm.missing_gain_fraction, float(stats.chunk.init_chi2), mean_chi)

        message = ("{}: {}, stall {:.2%}, g/fl {:.2%}, d/fl {:.2%}, "
                    "chi2 {:.4} -> {:.4}").format(*logvars)

        if sol_opts['last-rites']:

            logvars = (float(mean_chi1), float(stats.chunk.init_noise), float(stats.chunk.noise))

            message += " ({:.4}), noise {:.3} -> {:.3}".format(*logvars)
        
        print>> log, message

    # If everything has been flagged, no valid solutions are generated. 

    else:
        
        if isinstance(gm, jones_chain_machine.JonesChain):
            termstring = ""
            for term in gm.jones_terms:
                termstring += "{}: {} iters, ".format(term.jones_label, term.iters)
        else:
            termstring = "{} iters, ".format(gm.iters) 
        
        logvars = (label, termstring, n_gflags / float(gm.gflags.size), gm.missing_gain_fraction)

        print>>log, ModColor.Str("{} completely flagged after {} iters:"
                                 " g/fl {:.2%}, d/fl {:.2%}").format(*logvars)

        stats.chunk.chi2 = 0
        resid_arr = obser_arr

    stats.chunk.iters = gm.iters
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
        self.cmodel_arr = None
        self.freq_slice = freq_slice

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
                self._wobs_arr = np.empty_like(self.model_arr[0,...])
                self._wobs_arr[np.newaxis, ...] = self.obser_arr
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
            # if no weights, the simply duplicate the model axis in obser_arr
            else:
                self._wmod_arr = self.model_arr
        return self._wmod_arr

    @property
    def corrupt_weighted_model(self):
        """
        This property gives the model visibilities, corrupted by the gains, times the weights

        Returns:
            Weighted corrupted model visibilities (np.ndarray) of shape (n_dir, n_mod, n_tim, n_fre, n_ant, n_ant, n_cor, n_cor)
        """
        cmod = self.corrupt_model()
        if self.weight_arr is not None:
            return cmod*self.weight_arr[..., np.newaxis, np.newaxis]
        else:
            return cmod

    def corrupt_residual(self, imod=0):
        """
        This method returns the (corrupted) residual with respect to a given model
        
        Args:
            imod (int): 
                Index of model (0 to n_mod-1). 

        Returns:
            Weighted residual visibilities (np.ndarray) of shape (n_tim, n_fre, n_ant, n_ant, n_cor, n_cor)
        """
        if self.cmodel_arr is not None:
            return self.obser_arr - self.cmodel_arr[imod,...]
        else:
            resid_vis = np.empty_like(self.model_arr[0,0:1,...])
            self.gm.compute_residual(self.obser_arr, self.model_arr[:,0:1,...], resid_vis)
            return resid_vis[0,...]


    def corrupt_model(self, imod=None):
        """
        This method retuns the model visibilities, corrupted by the gains. If n_mod>1, then
        corrupt_model(None) must be called first (to corrupt all models). Note that this corrupts 
        the model array in place.

        Args:
            imod (int or None): 
                Index of model (0 to n_mod-1), or None to corrupt all models

        Returns:
            Corrupted model visibilities (np.ndarray) of shape (n_mod, n_tim, n_fre, n_ant, n_ant, n_cor, n_cor)
            if imod is None, otherwise the model axis is omitted.
        """
        # if asking for all models (imod=None), or we only have one model, then cache result
        if imod is None or self.model_arr.shape[1] == 1:
            if self.cmodel_arr is None:
                self.gm.apply_gains(self.model_arr)
                self.cmodel_arr = self.model_arr.sum(0)
            return self.cmodel_arr if imod is None else self.cmodel_arr[imod,...]
        # else just compute one particular model
        elif self.cmodel_arr is not None:
            return self.cmodel_arr[imod,...]
        else:
            return self.gm.apply_gains(self.model_arr[:, imod:imod+1, ...]).sum(0)


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
                                        sol_opts, label=label,
                                        compute_residuals=(vdm.weight_arr is None))

    # compute IFR gains, if needed. Note that this computes corrupt models, so it makes sense
    # doing it before recomputing the residuals: saves time
    if ifrgain_machine.is_computing():
        ifrgain_machine.update(vdm.weighted_obser, vdm.corrupt_weighted_model, vdm.flags_arr, vdm.freq_slice, soldict)

    # compute residuals if needed
    if vdm.weight_arr is not None:
        resid_vis = vdm.corrupt_residual(0)
    else:
        resid_vis = resid_vis[0, ...]

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

    resid_vis = vdm.corrupt_residual(0)

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


def run_solver(solver_type, itile, chunk_key, sol_opts):
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

    label = None
    
    try:
        tile = Tile.tile_list[itile]
        label = chunk_key
        solver = SOLVERS[solver_type]
        # initialize the gain machine for this chunk

        if gm_factory is None:
            raise RuntimeError("Gain machine factory has not been initialized")

        # Get chunk data from tile.

        obser_arr, model_arr, flags_arr, weight_arr = tile.get_chunk_cubes(chunk_key)

        chunk_ts, chunk_fs, _, freq_slice = tile.get_chunk_tfs(chunk_key)

        # apply IFR-based gains, if any
        ifrgain_machine.apply(obser_arr, freq_slice)

        # create subdict in shared dict for solutions etc.
        soldict = tile.create_solutions_chunk_dict(chunk_key)

        # create VisDataManager for this chunk

        vdm = _VisDataManager(obser_arr, model_arr, flags_arr, weight_arr, freq_slice)

        n_dir, n_mod = model_arr.shape[0:2] if model_arr is not None else (1,1)

        # create GainMachine
        vdm.gm = gm_factory.create_machine(vdm.weighted_obser, n_dir, n_mod, chunk_ts, chunk_fs)

        # Invoke solver method

        corr_vis, stats = solver(vdm, soldict, label, sol_opts)

        # Copy results back into tile.

        tile.set_chunk_cubes(corr_vis, flags_arr if (stats and stats.chunk.num_sol_flagged) else None, chunk_key)

        # Ask the gain machine to store its solutions in the shared dict.
        gm_factory.export_solutions(vdm.gm, soldict)

        return stats

    except Exception, exc:
        print>>log,ModColor.Str("Solver for tile {} chunk {} failed with exception: {}".format(itile, label, exc))
        print>>log,traceback.format_exc()
        raise

