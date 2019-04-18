# CubiCal: a radio interferometric calibration suite
# (c) 2017 Rhodes University & Jonathan S. Kenyon
# http://github.com/ratt-ru/CubiCal
# This code is distributed under the terms of GPLv2, see LICENSE.md for details
"""
Implements the solver loop.
"""
import numpy as np
import os, os.path
import traceback
from cubical.tools import logger, ModColor
from cubical.flagging import FL
from cubical.statistics import SolverStats
from cubical.tools import BREAK  # useful: can set static breakpoints by putting BREAK() in the code

## uncomment this to make UserWarnings (from e.g. numpy.ma) into full-blown exceptions
## TODO: add a --debug-catch-warnings option for this?
#import warnings
#warnings.simplefilter('error', UserWarning)
#warnings.simplefilter('error', RuntimeWarning)

from madmax.flagger import Flagger

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

# set to true for old-style (version <= 1.2.1) weight averaging, where 2x2 weights are collapsed into a single number
legacy_version12_weights = False

import __builtin__
try:
    __builtin__.profile
except AttributeError:
    # No line profiler, provide a pass-through version
    def profile(func): return func
    __builtin__.profile = profile


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
    chi_interval = sol_opts["chi-int"]
    stall_quorum = sol_opts["stall-quorum"]

    
    # collect flagging options

    flag_warning_threshold = GD['flags']["warn-thr"]
    
    # Initialise stat object.

    stats = SolverStats(obser_arr)
    stats.chunk.label = label
    stats.chunk.num_prior_flagged = (flags_arr&~(FL.MISSING|FL.SKIPSOL) != 0).sum()  # number of prior flagged data points
    stats.chunk.num_data_points = (flags_arr == 0).sum()                             # nominal number of valid data points

    diverging = ""

    # initialize iteration counter

    num_iter = 0

    # Estimates the overall noise level and the inverse variance per channel and per antenna as
    # noise varies across the band. This is used to normalize chi^2.

    stats.chunk.noise, inv_var_antchan, inv_var_ant, inv_var_chan = \
                                        stats.estimate_noise(obser_arr, flags_arr)

    # if we have directions in the model, but the gain machine is non-DD, collapse them
    if not gm.dd_term and model_arr.shape[0] > 1:
        model_arr = model_arr.sum(axis=0, keepdims=True)

    # This works out the conditioning of the solution, sets up various chi-sq normalization
    # factors etc, and does any other precomputation required by the current gain machine.

    gm.precompute_attributes(obser_arr, model_arr, flags_arr, inv_var_chan)

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

    # Initialize a residual array.

    resid_shape = [gm.n_mod, gm.n_tim, gm.n_fre, gm.n_ant, gm.n_ant, gm.n_cor, gm.n_cor]

    resid_arr = gm.cykernel.allocate_vis_array(resid_shape, obser_arr.dtype, zeros=True)
    gm.compute_residual(obser_arr, model_arr, resid_arr)
    resid_arr[:,flags_arr!=0] = 0

    # This flag is set to True when we have an up-to-date residual in resid_arr.
    
    have_residuals = True

    # apply MAD flagging
    madmax = Flagger(GD, label, metadata, stats)

    # do mad max flagging, if requested
    thr1, thr2 = madmax.get_mad_thresholds()
    if thr1 or thr2:
        if madmax.beyond_thunderdome(resid_arr, obser_arr, model_arr, flags_arr, thr1, thr2, "{} initial".format(label)):
            gm.update_equation_counts(flags_arr != 0)
            stats.chunk.num_mad_flagged = ((flags_arr & FL.MAD) != 0).sum()

    # In the event that there are no solutions with valid data, this will log some of the
    # flag information and break out of the function.
    stats.chunk.num_solutions = gm.num_solutions
    stats.chunk.num_sol_flagged = gm.num_gain_flags()[0]

    # every chunk stat set above now copied to stats.chunk.field_0
    stats.save_chunk_stats(step=0)

    if not gm.has_valid_solutions:
        print>> log, ModColor.Str("{} no solutions: {}; flags {}".format(label,
                        gm.conditioning_status_string, get_flagging_stats()))
        return (obser_arr if compute_residuals else None), stats, None

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

    chi, stats.chunk.chi2u = compute_chisq(statfield='initchi2')
    stats.chunk.chi2_0 = stats.chunk.chi2u_0 = stats.chunk.chi2u

    # The following provides conditioning information when verbose is set to > 0.
    if log.verbosity() > 0:

        print>> log, "{} chi^2_0 {:.4}; {}; noise {:.3}, flags: {}".format(
                        label, stats.chunk.chi2_0, gm.conditioning_status_string,
                        float(stats.chunk.noise_0), get_flagging_stats())

    # Main loop of the NNLS method. Terminates after quorum is reached in either converged or
    # stalled solutions or when the maximum number of iterations is exceeded.

    major_step = 0  # keeps track of "major" solution steps, for purposes of collecting stats

    while not(gm.has_converged) and not(gm.has_stalled):

        num_iter, update_major_step = gm.next_iteration()

        if update_major_step:
            major_step += 1
            stats.save_chunk_stats(step=major_step)

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

            stats.chunk.num_sol_flagged = gm.num_gain_flags()[0]

            # Adding the below lines for the robust solver so that flags should be apply to the weights
            if hasattr(gm, 'new_flags'):
                gm.new_flags = new_flags

        # print>>log,"{} {} {}".format(de.gains[1,5,2,5], de.posterior_gain_error[1,5,2,5], de.posterior_gain_error[1].mean())
        #
        have_residuals = False

        # Compute values used in convergence tests. This check implicitly marks flagged gains as 
        # converged.
        
        gm.check_convergence(gm.epsilon)

        stats.chunk.iters = num_iter
        stats.chunk.num_converged = gm.num_converged_solutions
        stats.chunk.frac_converged = gm.num_solutions and gm.num_converged_solutions / float(gm.num_solutions)

        # Break out of the solver loop if we find ourselves with no valid solution intervals (e.g. due to gain flagging)
        if not gm.has_valid_solutions:
            break

        # Check residual behaviour after a number of iterations equal to chi_interval. This is
        # expensive, so we do it as infrequently as possible.

        if (num_iter % chi_interval) == 0 or num_iter <= 1:

            old_chi, old_mean_chi = chi, float(stats.chunk.chi2u)

            gm.compute_residual(obser_arr, model_arr, resid_arr)
            resid_arr[:,flags_arr!=0] = 0

            # do mad max flagging, if requested
            thr1, thr2 = madmax.get_mad_thresholds()
            if thr1 or thr2:
                if madmax.beyond_thunderdome(resid_arr, obser_arr, model_arr, flags_arr, thr1, thr2,
                                             "{} iter {} ({})".format(label, num_iter, gm.jones_label)):
                    gm.update_equation_counts(flags_arr != 0)
                    stats.chunk.num_mad_flagged = ((flags_arr&FL.MAD) != 0).sum()

            chi, stats.chunk.chi2u = compute_chisq()

            have_residuals = True

            # Check for stalled solutions - solutions for which the residual is no longer improving.
            # Don't do this on a major step (i.e. when going from term to term in a chain), as the
            # reduced chisq (which compute_chisq() returns) can actually jump when going to the next term

            if update_major_step:
                stats.chunk.num_stalled = stats.chunk.num_diverged = 0
            else:
                delta_chi = old_chi - chi
                stats.chunk.num_stalled = np.sum((delta_chi <= gm.delta_chi*old_chi))
                stats.chunk.num_diverged = np.sum((delta_chi < -0.1 * old_chi))

            stats.chunk.frac_stalled = stats.chunk.num_stalled / float(chi.size)
            stats.chunk.frac_diverged = stats.chunk.num_diverged / float(chi.size)

            gm.has_stalled = (stats.chunk.frac_stalled >= stall_quorum)

            # if gm.has_stalled:
            #     import pdb; pdb.set_trace()

            if log.verbosity() > 1:
                if update_major_step:
                    delta_chi_max = delta_chi_mean = 0.
                else:
                    wh = old_chi != 0
                    delta_chi[wh] /= old_chi[wh]
                    delta_chi_max  = delta_chi.max()
                    delta_chi_mean = (old_mean_chi - stats.chunk.chi2u) / stats.chunk.chi2u

                if stats.chunk.num_diverged:
                    diverging = ", " + ModColor.Str("diverging {:.2%}".format(stats.chunk.frac_diverged), "red")
                else:
                    diverging = ""

                print>> log(2), ("{} {} chi2 {:.4}, rel delta {:.4} max {:.4}, active {:.2%}{}").format(
                                    label, gm.current_convergence_status_string,
                                    stats.chunk.chi2u, delta_chi_mean, delta_chi_max, float(1-stats.chunk.frac_stalled), diverging)

    # num_valid_solutions will go to 0 if all solution intervals were flagged. If this is not the
    # case, generate residuals etc.

    if gm.has_valid_solutions:
        # Final round of flagging
        flagged = gm.flag_solutions(flags_arr, True)
        stats.chunk.num_sol_flagged = gm.num_gain_flags()[0]
    else:
        flagged = None
        
    # check this again, because final round of flagging could have killed us
    if gm.has_valid_solutions:
        # Do we need to recompute the final residuals?
        if (sol_opts['last-rites'] or compute_residuals) and (not have_residuals or flagged):
            gm.compute_residual(obser_arr, model_arr, resid_arr)
            resid_arr[:,flags_arr!=0] = 0

            # do mad max flagging, if requested
            thr1, thr2 = madmax.get_mad_thresholds()
            if thr1 or thr2:
                if madmax.beyond_thunderdome(resid_arr, obser_arr, model_arr, flags_arr, thr1, thr2,
                                            "{} final".format(label)) and sol_opts['last-rites']:
                    gm.update_equation_counts(flags_arr != 0)

        # Re-estimate the noise using the final residuals, if last rites are needed.

        if sol_opts['last-rites']:
            stats.chunk.noise, inv_var_antchan, inv_var_ant, inv_var_chan = \
                                        stats.estimate_noise(resid_arr, flags_arr, residuals=True)
            chi1, stats.chunk.chi2 = compute_chisq(statfield='chi2')
        else:
            stats.chunk.chi2 = stats.chunk.chi2u

        message = "{} (end solve) {}, stall {:.2%}{}, chi^2 {:.4} -> {:.4}".format(label, gm.final_convergence_status_string,
                    float(stats.chunk.frac_stalled), diverging, float(stats.chunk.chi2_0), stats.chunk.chi2u)

        if sol_opts['last-rites']:

            message = "{} ({:.4}), noise {:.3} -> {:.3}".format(message,
                            float(stats.chunk.chi2), float(stats.chunk.noise_0), float(stats.chunk.noise))

        print>> log, message

    # If everything has been flagged, no valid solutions are generated. 

    else:
        
        print>>log(0, "red"), "{} (end solve) {}: completely flagged?".format(label, gm.final_convergence_status_string)

        chi2 = chi2u = 0
        resid_arr = obser_arr

    # collect messages from various flagging sources, and print to log if any
    flagstatus = []

    if stats.chunk.num_sol_flagged:
        # also for up message with flagging stats
        fstats = []
        for flagname, mask in FL.categories().iteritems():
            if mask != FL.MISSING:
                n_flag, n_tot = gm.num_gain_flags(mask)
                if n_flag:
                    fstats.append("{}:{}({:.2%})".format(flagname, n_flag, n_flag/float(n_tot)))

        flagstatus.append("solver flags {}".format(" ".join(fstats)))

    if stats.chunk.num_mad_flagged:
        flagstatus.append("{} took out {} visibilities".format(madmax.desc_mode, stats.chunk.num_mad_flagged))

    if flagstatus:
        # clear Mad Max flags if in trial mode
        if madmax.trial_mode:
            flags_arr &= ~FL.MAD
        n_new_flags = (flags_arr&~(FL.MISSING|FL.SKIPSOL) != 0).sum() - stats.chunk.num_prior_flagged
        if n_new_flags < flags_arr.size*flag_warning_threshold:
            warning, color = "", "blue"
        else:
            warning, color = "WARNING: ", "red"
        print>> log(0, color), "{}{} {}: {} ({:.2%}) new data flags".format(
            warning, label, ", ".join(flagstatus),
            n_new_flags, n_new_flags / float(flags_arr.size))

    robust_weights = None
    if hasattr(gm, 'save_weights'):
        if gm.save_weights:
            newshape = gm.weights.shape[1:-1] + (2,2)
            robust_weights = np.repeat(gm.weights.real, 4, axis=-1)
            robust_weights = np.reshape(robust_weights, newshape) 
 
    
    return (resid_arr if compute_residuals else None), stats, robust_weights


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
        ## OMS: take sqrt() of weights since that's the correct thing to use in whitening
        self.obser_arr, self.model_arr, self.flags_arr, self.weight_arr = \
            obser_arr, model_arr, flags_arr, weight_arr
        if legacy_version12_weights:
            # self.weight_arr[:] = np.sqrt(self.weight_arr.mean(axis=(-1,-2)))[..., np.newaxis, np.newaxis]
            self.weight_arr[:] = self.weight_arr.mean(axis=(-1,-2))[..., np.newaxis, np.newaxis]
        else:
            np.sqrt(self.weight_arr, out=self.weight_arr)
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
                self._wobs_arr = self.obser_arr[np.newaxis,...] * self.weight_arr
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
                self._wmod_arr = self.model_arr * self.weight_arr[np.newaxis, ...]
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
            return cmod*self.weight_arr
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

    _, stats, outweights = _solve_gains(vdm.gm, vdm.weighted_obser, vdm.weighted_model, vdm.flags_arr, sol_opts, label=label)
    if ifrgain_machine.is_computing():
        ifrgain_machine.update(vdm.weighted_obser, vdm.corrupt_weighted_model, vdm.flags_arr, vdm.freq_slice, soldict)

    return None, stats, outweights


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

    _, stats, outweights = _solve_gains(vdm.gm, vdm.weighted_obser, vdm.weighted_model, vdm.flags_arr, sol_opts, label=label)

    # for corrected visibilities, take the first data/model pair only
    corr_vis = np.zeros_like(vdm.obser_arr)
    vdm.gm.apply_inv_gains(vdm.obser_arr, corr_vis)

    if ifrgain_machine.is_computing():
        ifrgain_machine.update(vdm.weighted_obser, vdm.corrupt_weighted_model, vdm.flags_arr, vdm.freq_slice, soldict)

    return corr_vis, stats, outweights


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
    resid_vis, stats, outweights = _solve_gains(vdm.gm, vdm.weighted_obser, vdm.weighted_model, vdm.flags_arr,
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
        return corr_vis, stats, outweights
    else:
        return resid_vis, stats, outweights

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

    return corr_vis, None, None


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
        return corr_vis, None, None
    else:
        return resid_vis, None, None

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

        corr_vis, stats, outweights = solver(vdm, soldict, label, sol_opts)
        
        # Panic if amplitude has gone crazy
        
        if debug_opts['panic-amplitude']:
            if corr_vis is not None:
                unflagged = flags_arr==0
                if unflagged.any() and abs(corr_vis[unflagged,:,:]).max() > debug_opts['panic-amplitude']:
                    raise RuntimeError("excessive amplitude in chunk {}".format(label))

        # Copy results back into tile.
        have_new_flags = stats and ( stats.chunk.num_sol_flagged > 0 or stats.chunk.num_mad_flagged > 0)

        tile.set_chunk_cubes(corr_vis, flags_arr if have_new_flags else None, outweights, chunk_key)

        # Ask the gain machine to store its solutions in the shared dict.
        gm_factory.export_solutions(vdm.gm, soldict)

        return stats

    except Exception, exc:
        print>>log,ModColor.Str("Solver for tile {} chunk {} failed with exception: {}".format(itile, label, exc))
        print>>log,traceback.format_exc()
        raise

