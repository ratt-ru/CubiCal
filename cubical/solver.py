# CubiCal: a radio interferometric calibration suite
# (c) 2017 Rhodes University & Jonathan S. Kenyon
# http://github.com/ratt-ru/CubiCal
# This code is distributed under the terms of GPLv2, see LICENSE.md for details
"""
Implements the solver loop.
"""
from __future__ import print_function
import numpy as np
import os, os.path
import gc
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

from .madmax.flagger import Flagger

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

import builtins
try:
    builtins.profile
except AttributeError:
    # No line profiler, provide a pass-through version
    def profile(func): return func
    builtins.profile = profile

@builtins.profile
def _solve_gains(gm, stats, madmax, obser_arr, model_arr, flags_arr, sol_opts, label="", compute_residuals=None):
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

    diverging = ""

    # for all the solvers that do not output any weights and for the robust solver when they are no valid solutions
    gm.output_weights = None 

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

    # apply any flags raised in the precompute

    new_flags = flags_arr & ~(FL.MISSING | FL.PRIOR) != 0
    model_arr[:, :, new_flags, :, :] = 0
    obser_arr[:, new_flags, :, :] = 0

    def get_flagging_stats():
        """Returns a string describing per-flagset statistics"""
        fstats = []

        for flag, mask in FL.categories().items():
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

    resid_arr = gm.allocate_vis_array(resid_shape, obser_arr.dtype, zeros=True)
    gm.compute_residual(obser_arr, model_arr, resid_arr, require_full=True)
    resid_arr[:,flags_arr!=0] = 0

    # apply MAD flagging
    madmax.set_mode(GD['madmax']['enable'])

    # do mad max flagging, if requested
    thr1, thr2 = madmax.get_mad_thresholds()
    if thr1 or thr2:
        if madmax.beyond_thunderdome(resid_arr, obser_arr, model_arr, flags_arr, thr1, thr2, "{} initial".format(label)):
            gm.update_equation_counts(flags_arr != 0)
            stats.chunk.num_mad_flagged = ((flags_arr & FL.MAD) != 0).sum()


    # apply robust flag if robust machine (this uses the madmax flag)
    if hasattr(gm, 'is_robust'):
        if gm.robust_flag_weights:
            gm.robust_flag(flags_arr, model_arr, obser_arr)
            stats.chunk.num_mad_flagged = ((flags_arr & FL.MAD) != 0).sum()

    # In the event that there are no solutions with valid data, this will log some of the
    # flag information and break out of the function.
    stats.chunk.num_solutions = gm.num_solutions
    stats.chunk.num_sol_flagged = gm.num_gain_flags()[0]

    # every chunk stat set above now copied to stats.chunk.field_0
    stats.save_chunk_stats(step=0)

    # raise warnings from priori conditioning, before the loop
    for d in gm.collect_warnings():
        log.write(d["msg"],
                  level=d["level"],
                  print_once=d["raise_once"],
                  verbosity=d["verbosity"],
                  color=d["color"])

    if not gm.has_valid_solutions:
        log.error("{} no solutions: {}; flags {}".format(label, gm.conditioning_status_string, get_flagging_stats()))
        return (obser_arr if compute_residuals else None), stats, None

    def compute_chisq(statfield=None, full=True):
        """
        Computes chi-squared statistic based on current residuals and noise estimates.
        Populates the stats object with it.

        Full=True at the beginning and end of a solution, and it is passed to gm.compute_chisq()
        """
        chisq, chisq_per_tf_slot, chisq_tot = gm.compute_chisq(resid_arr, inv_var_chan, require_full=full)

        if statfield:
            getattr(stats.chanant, statfield)[...]  = np.sum(chisq, axis=0)
            getattr(stats.timeant, statfield)[...]  = np.sum(chisq, axis=1)
            getattr(stats.timechan, statfield)[...] = np.sum(chisq, axis=2)

        return chisq_per_tf_slot, chisq_tot

    chi, stats.chunk.chi2u = compute_chisq(statfield='initchi2', full=True)
    stats.chunk.chi2_0 = stats.chunk.chi2u_0 = stats.chunk.chi2u

    # The following provides conditioning information when verbose is set to > 0.
    if log.verbosity() > 0:
        log(1).print("{} chi^2_0 {:.4}; {}; noise {:.3}, flags: {}".format(
                        label, stats.chunk.chi2_0, gm.conditioning_status_string,
                        float(stats.chunk.noise_0), get_flagging_stats()))

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

            gm.compute_residual(obser_arr, model_arr, resid_arr, require_full=False)
            resid_arr[:,flags_arr!=0] = 0

            # do mad max flagging, if requested
            thr1, thr2 = madmax.get_mad_thresholds()
            if thr1 or thr2:
                num_mad_flagged_prior = int(stats.chunk.num_mad_flagged)

                if madmax.beyond_thunderdome(resid_arr, obser_arr, model_arr, flags_arr, thr1, thr2,
                                             "{} iter {} ({})".format(label, num_iter, gm.jones_label)):
                    gm.update_equation_counts(flags_arr != 0)
                    stats.chunk.num_mad_flagged = ((flags_arr&FL.MAD) != 0).sum()
                    if stats.chunk.num_mad_flagged != num_mad_flagged_prior:
                        log(2).print("{}: {} new MadMax flags".format(label,
                                        stats.chunk.num_mad_flagged - num_mad_flagged_prior))

            chi, stats.chunk.chi2u = compute_chisq(full=False)

            # Check for stalled solutions - solutions for which the residual is no longer improving.
            # Don't do this on a major step (i.e. when going from term to term in a chain), as the
            # reduced chisq (which compute_chisq() returns) can actually jump when going to the next term

            if update_major_step:
                stats.chunk.num_stalled = stats.chunk.num_diverged = 0
            else:
                delta_chi = old_chi - chi
                stats.chunk.num_stalled = np.sum((delta_chi <= gm.delta_chi*old_chi))
                diverged_tf_slots = delta_chi < -0.1 * old_chi
                stats.chunk.num_diverged = diverged_tf_slots.sum()
                # at first iteration, flag immediate divergers
                if sol_opts['flag-divergence'] and stats.chunk.num_diverged and num_iter == 1:
                    model_arr[:, :, diverged_tf_slots] = 0
                    obser_arr[:, diverged_tf_slots] = 0

                    # find previously unflagged visibilities that have become flagged due to divergence
                    new_flags = (flags_arr == 0)
                    new_flags[~diverged_tf_slots] = 0

                    flags_arr[diverged_tf_slots] |= FL.DIVERGE

                    num_nf = new_flags.sum()
                    log.warn("{}: {:.2%} slots diverging, {} new data flags".format(label,
                                    diverged_tf_slots.sum()/float(diverged_tf_slots.size), num_nf))

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
                    chi_mean = float(stats.chunk.chi2u)
                    delta_chi_mean = (old_mean_chi - chi_mean) / chi_mean if chi_mean != 0 else 0.

                if stats.chunk.num_diverged:
                    diverging = ", " + ModColor.Str("diverging {:.2%}".format(stats.chunk.frac_diverged), "red")
                else:
                    diverging = ""

                log(2).print("{} {} chi2 {:.4}, rel delta {:.4} max {:.4}, active {:.2%}{}".format(
                                    label, gm.current_convergence_status_string,
                                    stats.chunk.chi2u, delta_chi_mean, delta_chi_max,
                                    float(1-stats.chunk.frac_stalled), diverging))

        # Adding the below lines for the robust solver so that flags should be apply to the weights
        if hasattr(gm, 'is_robust'):
            gm.update_weight_flags(flags_arr)


    # num_valid_solutions will go to 0 if all solution intervals were flagged. If this is not the
    # case, generate residuals etc.

    if gm.has_valid_solutions:
        # Final round of flagging
        flagged = gm.flag_solutions(flags_arr, final=True)
        stats.chunk.num_sol_flagged = gm.num_gain_flags(final=True)[0]
    else:
        flagged = None



    # check this again, because final round of flagging could have killed us
    if gm.has_valid_solutions:
        # Do we need to recompute the final residuals?
        if (sol_opts['last-rites'] or compute_residuals):
            gm.compute_residual(obser_arr, model_arr, resid_arr, require_full=True)
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
            chi1, stats.chunk.chi2 = compute_chisq(statfield='chi2', full=True)
        else:
            stats.chunk.chi2 = stats.chunk.chi2u

        message = "{} (end solve) {}, stall {:.2%}{}, chi^2 {:.4} -> {:.4}".format(label, gm.final_convergence_status_string,
                  float(stats.chunk.frac_stalled), diverging, float(stats.chunk.chi2_0), stats.chunk.chi2u)

        should_warn = float(stats.chunk.chi2_0) < float(stats.chunk.chi2u) or diverging
        if sol_opts['last-rites'] and (should_warn or log.verbosity() > 0):
            message = "{} ({:.4}), noise {:.3} -> {:.3}".format(message,
                            float(stats.chunk.chi2), float(stats.chunk.noise_0), float(stats.chunk.noise))
        if should_warn:
            message += " Shows signs of divergence. If you see this message often you may have significant RFI present in your data or your solution intervals are too short."
        if should_warn:
            log.warn(message)
        elif log.verbosity() > 0:
            log.info(message)

    # If everything has been flagged, no valid solutions are generated.

    else:
        log.error("{} (end solve) {}: completely flagged?".format(label, gm.final_convergence_status_string))

        chi2 = chi2u = 0
        resid_arr = obser_arr

    robust_weights = None
    if hasattr(gm, 'is_robust'):
        
        # do a last round of robust flag robust flag and save the weights
         
        if gm.robust_flag_weights and not gm.robust_flag_disable:
            gm.robust_flag(flags_arr, model_arr, obser_arr, final=True)
            stats.chunk.num_mad_flagged = ((flags_arr & FL.MAD) != 0).sum()
        
        if gm.save_weights:
            newshape = gm.weights.shape[1:-1] + (2,2)
            robust_weights = np.repeat(gm.weights.real, 4, axis=-1)
            robust_weights = np.reshape(robust_weights, newshape)
            gm.output_weights = robust_weights
        

    # After the solver loop check for warnings from the solvers
    for d in gm.collect_warnings():
        log.write(d["msg"],
                  level=d["level"],
                  print_once=d["raise_once"],
                  verbosity=d["verbosity"],
                  color=d["color"])


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
        if GD['data']['normalize']:
            self._normalization = np.abs(obser_arr)
        else:
            self._normalization = None
        if weight_arr is not None:
            if legacy_version12_weights:
                # self.weight_arr[:] = np.sqrt(self.weight_arr.mean(axis=(-1,-2)))[..., np.newaxis, np.newaxis]
                if self._normalization is not None:
                    self.weight_arr = weight_arr*self._normalization
                self.weight_arr[:] = self.weight_arr.mean(axis=(-1,-2))[..., np.newaxis, np.newaxis]
            else:
                np.sqrt(self.weight_arr, out=self.weight_arr)
                if self._normalization is not None:
                    self.weight_arr *= self._normalization
            
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
            if self._normalization is not None:
                with np.errstate(divide='ignore', invalid='ignore'):
                    self._wobs_arr /= self._normalization[np.newaxis,...]
                    self._wobs_arr[(self._normalization==0)[np.newaxis,...]] = 0
                self._normalization = None
              
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
            self.gm.apply_gains(self.model_arr, full2x2=True)
            self._model_corrupted = True
        if imod is None:
            return self.model_arr
        return self.model_arr[idir, imod].sum(0)


class classproperty(property):
    def __get__(self, cls, owner):
        return classmethod(self.fget).__get__(None, owner)()

class SolverMachine(object):
    """Base class encapsulating different solver methods and their properties"""
    def __init__(self, vdm, gm, soldict, sol_opts, label, metadata):
        """
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
            soldict (:obj:~cubical.tools.shared_dict.SharedDict):
                Shared dict used to pass solutions (IFR gain solutions, primarily) back out to the
                calling thread.
        """
        self.sol_opts = sol_opts
        self.label = label
        self.metadata = metadata
        self.vdm = vdm
        self.gm  = gm
        self.soldict = soldict
        self.stats = SolverStats(vdm.obser_arr)
        self.stats.chunk.label = label
        self.stats.chunk.num_prior_flagged = (
                    vdm.flags_arr & ~(FL.MISSING | FL.SKIPSOL) != 0).sum()  # number of prior flagged data points
        self.stats.chunk.num_data_points = (vdm.flags_arr == 0).sum()  # nominal number of valid data points

        # for apply-only machines, precompute machine attributes and apply initial gain flags
        if self.is_apply_only:
            gm.precompute_attributes(vdm.obser_arr, vdm.model_arr, vdm.flags_arr, None)
            gm.flag_solutions(vdm.flags_arr, True)
            self.stats.chunk.num_solutions = vdm.gm.num_solutions
            self.stats.chunk.num_sol_flagged = vdm.gm.num_gain_flags(final=True)[0]

        # initialize the flagger
        self.madmax = Flagger(GD, label, metadata, self.stats)
        self.madmax.set_mode(GD['madmax']['enable'])

        # output weights, if any, go here
        self.output_weights = None

    # various traits of the solver machine, redefined by subclasses

    # does this machine require model visibilities?
    @classproperty
    def is_model_required(cls):
        return True

    # does this machine produce full corrected residuals?
    @property
    def outputs_full_corrected_residuals(cls):
        return False

    # is this an apply-only (i.e. non-solver) machine?
    is_apply_only = False


    def run(self):
        """
        Abstract method to run the solver.

        Returns:
            np.array of output visibilities
        """
        return NotImplementedError

    def finalize(self, corr_vis):
        """
        Finalizes the output visibilities, running a pass of the flagger on them, if configured
        """
        
        # clear out MAD flags if madmax was in trial mode
        if self.stats.chunk.num_mad_flagged and self.madmax.trial_mode:
            self.vdm.flags_arr &= ~FL.MAD
            self.stats.chunk.num_mad_flagged = 0
        num_mad_flagged_prior = int(self.stats.chunk.num_mad_flagged)

        # apply final round of madmax on residuals, if asked to
        if GD['madmax']['residuals']:
            # recompute the residuals if required
            if self.outputs_full_corrected_residuals:
                resid_vis = corr_vis
                log(0).print("{}: doing final MadMax round on residuals".format(self.label))
            else:
                log(0).print("{}: computing full residuals for final MadMax round".format(self.label))
                resid_vis1 = self.vdm.corrupt_residual(self.sol_opts["subtract-model"], slice(None))
                resid_vis = np.zeros_like(resid_vis1)
                self.gm.apply_inv_gains(resid_vis1, resid_vis, full2x2=True,
                                        direction=self.sol_opts["correct-dir"])
                del resid_vis1

            # clear the SKIPSOL flag to also flag data that's been omitted from the solutions
            self.vdm.flags_arr &= ~FL.SKIPSOL

            # run madmax on them
            self.madmax.set_mode(GD['madmax']['residuals'])
            thr1, thr2 = self.madmax.get_mad_thresholds()
            if thr1 or thr2:
                if self.madmax.beyond_thunderdome(resid_vis, None, None, self.vdm.flags_arr, thr1, thr2,
                                             "{} residual".format(self.label)):
                    self.stats.chunk.num_mad_flagged = ((self.vdm.flags_arr & FL.MAD) != 0).sum()
            resid_vis = None  # release memory if new object was created

        # collect messages from various flagging sources, and print to log if any
        flagstatus = []

        if self.stats.chunk.num_sol_flagged:
            # also for up message with flagging stats
            fstats = []
            for flagname, mask in FL.categories().items():
                if mask != FL.MISSING:
                    n_flag, n_tot = self.gm.num_gain_flags(mask, final=True)
                    if n_flag:
                        fstats.append("{}:{}({:.2%})".format(flagname, n_flag, n_flag/float(n_tot)))

            nfl, nsol = self.gm.num_gain_flags(final=True)
            flagstatus.append("gain flags {} ({:.2%} total)".format(" ".join(fstats), nfl/float(nsol)))

        if self.stats.chunk.num_mad_flagged:
            flagstatus.append("MadMax took out {} visibilities ({} in final round)".format(
                        self.stats.chunk.num_mad_flagged, self.stats.chunk.num_mad_flagged - num_mad_flagged_prior))

        if flagstatus:
            n_new_flags = (self.vdm.flags_arr&~(FL.MISSING|FL.SKIPSOL) != 0).sum() - self.stats.chunk.num_prior_flagged
            if n_new_flags < self.vdm.flags_arr.size*GD['flags']['warn-thr']:
                warning, color = "", "blue"
            else:
                warning, color = "", "red"
            log(0, color).print("{}{} has {} ({:.2%}) new data flags: {}".format(
                warning, self.label,
                n_new_flags, n_new_flags / float(self.vdm.flags_arr.size),
                ", ".join(flagstatus)))


class SolveOnly(SolverMachine):
    """Runs the solver, but does not apply solutions"""

    def run(self):
#        import ipdb; ipdb.set_trace()

        _solve_gains(self.gm, self.stats, self.madmax,
                     self.vdm.weighted_obser, self.vdm.weighted_model, self.vdm.flags_arr,
                     self.sol_opts, label=self.label)

        self.output_weights = self.gm.output_weights

        if ifrgain_machine.is_computing():
            ifrgain_machine.update(self.vdm.weighted_obser, self.vdm.corrupt_weighted_model, self.vdm.flags_arr,
                                   self.vdm.freq_slice, self.soldict)

        return None



class SolveAndCorrect(SolveOnly):
    """
    Run the solver and save and apply the resulting gain solutions to the observed data. Produces
    corrected data.
    """
    def run(self):
        SolveOnly.run(self)
        # for corrected visibilities, take the first data/model pair only
        corr_vis = np.zeros_like(self.vdm.obser_arr)
        self.gm.apply_inv_gains(self.vdm.obser_arr, corr_vis, full2x2=True, direction=self.sol_opts["correct-dir"])
        return corr_vis

class SolveAndSubtract(SolveOnly):
    """
    Run the solver, generate residuals, and (optionally) apply the resulting gain solutions to the residuals.
    Produces (un)corrected residuals.
    """
    # The SolveAndCorrectResiduals subclass redefines this to True
    output_corrected_residuals = False

    def run(self, correct=False):
        SolveOnly.run(self)
        # compute residuals
        resid_vis = self.vdm.corrupt_residual(self.sol_opts["subtract-model"],  self.sol_opts["subtract-dirs"])

        # correct residual if required
        if self.output_corrected_residuals:
            corr_vis = np.zeros_like(resid_vis)
            self.gm.apply_inv_gains(resid_vis, corr_vis, full2x2=True, direction=self.sol_opts["correct-dir"])
            return corr_vis
        else:
            return resid_vis

class SolveAndCorrectResiduals(SolveAndSubtract):
    """
    Run the solver, generate corrected residuals.
    """
    # mark this machine as generating corrected residuals
    output_corrected_residuals = True
    # mark this machine as generating full corrected residuals, if all directions are subtracted
    @property
    def outputs_full_corrected_residuals(self):
        return self.sol_opts['subtract-dirs'] == slice(None)

class CorrectOnly(SolverMachine):
    """
    Do not solve. Apply prior gain solutions to the observed data, generating corrected data.
    """
    # mark machine as an apply-only type
    is_apply_only = True
    # model not required, unless we're flagging on residuals
    @classproperty
    def is_model_required(cls):
        return bool(GD['madmax']['residuals'])

    def run(self):
        if self.vdm.model_arr is not None and ifrgain_machine.is_computing():
            ifrgain_machine.update(self.vdm.weighted_obser, self.vdm.corrupt_weighted_model, self.vdm.flags_arr,
                                   self.vdm.freq_slice, self.soldict)

        corr_vis = np.zeros_like(self.vdm.obser_arr)
        self.gm.apply_inv_gains(self.vdm.obser_arr, corr_vis, full2x2=True, direction=self.sol_opts["correct-dir"])

        return corr_vis

class SubtractOnly(SolverMachine):
    """
    Do not solve. Apply prior gain solutions, generate (un)corrected residuals.
    The optional flag to run is invoked by the subclass
    """
    # mark machine as an apply-only type
    is_apply_only = True
    # The SubtractAndCorrect subclass redefines this to True
    output_corrected_residuals = False

    def run(self):
        # doing it before recomputing the residuals: saves time
        if ifrgain_machine.is_computing():
            ifrgain_machine.update(self.vdm.weighted_obser, self.vdm.corrupt_weighted_model, self.vdm.flags_arr,
                                   self.vdm.freq_slice, self.soldict)

        resid_vis = self.vdm.corrupt_residual(self.sol_opts["subtract-model"],  self.sol_opts["subtract-dirs"])

        # correct residual if required
        if self.output_corrected_residuals:
            corr_vis = np.zeros_like(resid_vis)
            self.gm.apply_inv_gains(resid_vis, corr_vis, full2x2=True, direction=self.sol_opts["correct-dir"])
            return corr_vis
        else:
            return resid_vis

class CorrectResiduals(SubtractOnly):
    """
    Do not solve. Apply prior gain solutions, generate corrected residuals.
    """
    # mark this machine as generating corrected residuals
    output_corrected_residuals = True
    # mark this machine as generating full corrected residuals, if all directions are subtracted
    @property
    def outputs_full_corrected_residuals(self):
        return self.sol_opts['subtract-dirs'] == slice(None)


SOLVERS = { 'so': SolveOnly,
            'sc': SolveAndCorrect,
            'sr': SolveAndCorrectResiduals,
            'ss': SolveAndSubtract,
            'ac': CorrectOnly,
            'ar': CorrectResiduals,
            'as': SubtractOnly
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
        allocate_vis_array, allocate_flag_array, _ = gm_factory.determine_allocators()

        obser_arr, model_arr, flags_arr, weight_arr = tile.get_chunk_cubes(chunk_key,
                                 gm_factory.ctype,
                                 allocator=allocate_vis_array,
                                 flag_allocator=allocate_flag_array)

        chunk_ts, chunk_fs, _, freq_slice = tile.get_chunk_tfs(chunk_key)

        # apply IFR-based gains, if any
        ifrgain_machine.apply(obser_arr, freq_slice)

        # create subdict in shared dict for solutions etc.
        soldict = tile.create_solutions_chunk_dict(chunk_key)

        # create VisDataManager for this chunk

        vdm = _VisDataManager(obser_arr, model_arr, flags_arr, weight_arr, freq_slice)

        n_dir, n_mod = model_arr.shape[0:2] if model_arr is not None else (1,1)

        solver_machine_class = SOLVERS[solver_type]
        # create GainMachine
        # import pudb; pu.db
        gm = vdm.gm = gm_factory.create_machine(vdm.weighted_obser, n_dir, n_mod, chunk_ts, chunk_fs, label)

        # create solver machine
        solver_machine = solver_machine_class(vdm, gm, soldict, sol_opts, label, metadata)

        # Invoke solver method
        if debug_opts['stop-before-solver']:
            import pdb
            pdb.set_trace()

        corr_vis = solver_machine.run()
        
        # Panic if amplitude has gone crazy

        if debug_opts['panic-amplitude']:
            if corr_vis is not None:
                unflagged = flags_arr==0
                if unflagged.any() and abs(corr_vis[unflagged,:,:]).max() > debug_opts['panic-amplitude']:
                    raise RuntimeError("excessive amplitude in chunk {}".format(label))

        # finalize residuals
        solver_machine.finalize(corr_vis)

        # Copy results back into tile.
        have_new_flags = (solver_machine.stats.chunk.num_sol_flagged > 0 or
                          solver_machine.stats.chunk.num_mad_flagged > 0)

        tile.set_chunk_cubes(corr_vis, flags_arr if have_new_flags else None,
                             solver_machine.output_weights, chunk_key)

        # Ask the gain machine to store its solutions in the shared dict.
        gm_factory.export_solutions(gm, soldict)

        # Trigger garbage collection because it seems very unreliable. This 
        # flattens the memory profile substantially. 
        gc.collect()

        return solver_machine.stats

    except Exception as exc:
        log.error("Solver for tile {} chunk {} failed with exception: {}".format(itile, label, exc))
        log.print(traceback.format_exc())
        raise

