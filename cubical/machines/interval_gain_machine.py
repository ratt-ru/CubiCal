# CubiCal: a radio interferometric calibration suite
# (c) 2017 Rhodes University & Jonathan S. Kenyon
# http://github.com/ratt-ru/CubiCal
# This code is distributed under the terms of GPLv2, see LICENSE.md for details
import numpy as np
from cubical.flagging import FL
from cubical.machines.abstract_machine import MasterMachine

from numpy.ma import masked_array

def copy_or_identity(array, time_ind=0, out=None):
    if out is None:
        return array
    np.copyto(out, array)
    return out


class PerIntervalGains(MasterMachine):
    """
    This is a base class for all gain solution machines that use solutions intervals.
    """

    def __init__(self, label, data_arr, ndir, nmod, times, frequencies, chunk_label, options, cykernel):
        """
        Initialises a gain machine which supports solution intervals.
        
        Args:
            label (str):
                Label identifying the Jones term.
            data_arr (np.ndarray): 
                Shape (n_mod, n_tim, n_fre, n_ant, n_ant, n_cor, n_cor) array containing observed 
                visibilities. 
            ndir (int):
                Number of directions.
            nmod (nmod):
                Number of models.
            times (np.ndarray):
                Times for the data being processed.
            freqs (np.ndarray):
                Frequencies for the data being processsed.
            options (dict): 
                Dictionary of options. 
        """

        MasterMachine.__init__(self, label, data_arr, ndir, nmod, times, frequencies,
                               chunk_label, options)

        self.cykernel = cykernel

        self.t_int = options["time-int"] or self.n_tim
        self.f_int = options["freq-int"] or self.n_fre
        self.eps = 1e-6

        # Initialise attributes used for computing values over intervals.
        # n_tim and n_fre are the time and frequency dimensions of the data arrays.
        # n_timint and n_freint are the time and frequency dimensions of the gains.

        self.t_bins = range(0, self.n_tim, self.t_int)
        self.f_bins = range(0, self.n_fre, self.f_int)

        self.n_timint = len(self.t_bins)
        self.n_freint = len(self.f_bins)
        self.n_tf_ints = self.n_timint * self.n_freint

        # number of valid solutions
        self.n_valid_sols = self.n_dir * self.n_tf_ints

        # split grids into intervals, and find the centre of gravity of each
        timebins = np.split(times, self.t_bins[1:])
        freqbins = np.split(frequencies, self.f_bins[1:])
        timegrid = np.array([float(x.mean()) for x in timebins])
        freqgrid = np.array([float(x.mean()) for x in freqbins])

        # interval_grid determines the per-interval grid poins
        self.interval_grid = dict(time=timegrid, freq=freqgrid)
        # data_grid determines the full resolution grid
        self.data_grid = dict(time=times, freq=frequencies)

        # compute index from each data point to interval number
        t_ind = np.arange(self.n_tim)//self.t_int
        f_ind = np.arange(self.n_fre)//self.f_int

        self.t_mapping, self.f_mapping = np.meshgrid(t_ind, f_ind, indexing="ij")

        # Initialise attributes used in convergence testing. n_cnvgd is the number
        # of solutions which have converged.

        self._has_stalled = False
        self.n_cnvgd = 0
        self._frac_cnvgd = 0
        self.iters = 0
        self.min_quorum = options["conv-quorum"]
        self.update_type = options["update-type"]
        self.ref_ant = options["ref-ant"]
        self.fix_directions = options["fix-dirs"] or []
        if type(self.fix_directions) is int:
            self.fix_directions = [self.fix_directions]
        # True if gains are loaded from a DB
        self._gains_loaded = False

        # Construct flag array and populate flagging attributes.
        self.max_gain_error = options["max-prior-error"]
        self.max_post_error = options["max-post-error"]

        self.clip_lower = options["clip-low"]
        self.clip_upper = options["clip-high"]
        self.clip_after = options["clip-after"]

        self.init_gains()
        self.old_gains = self.gains.copy()

        # Gain error estimates. Populated by subclasses, if available
        # Should be array of same shape as the gains
        self.prior_gain_error = None
        self.posterior_gain_error = None

        # buffers for arrays used in internal updates
        self._jh = self._jhr = self._jhj = self._gh = self._r = self._ginv = self._ghinv = None
        self._update = None

        # flag: have gains been updated
        self._gh_update = self._ghinv_update = True

    def get_conj_gains(self):
        if self._gh is None:
            self._gh = np.empty_like(self.gains)
        if self._gh_update:
            np.conj(self.gains.transpose(0, 1, 2, 3, 5, 4), out=self._gh)
            self._gh_update = False
        return self._gh

    def get_inverse_gains(self):
        if self._ginv is None:
            self._ginv = np.empty_like(self.gains)
            self._ghinv = np.empty_like(self.gains)
        if self._ghinv_update:
            self._ghinv_flag_count = self.cykernel.cyinvert_gains(
                self.gains, self._ginv, self.gflags, self.eps, FL.ILLCOND)
            np.conj(self._ginv.transpose(0, 1, 2, 3, 5, 4), out=self._ghinv)
            self._ghinv_update = False
        return self._ginv, self._ghinv, self._ghinv_flag_count

    def get_new_jh(self, model_arr):
        if self._jh is None:
            self._jh = np.empty_like(model_arr)
        # else:
        #     self._jh.fill(0)
        return self._jh

    def get_new_jhr(self):
        if self._jhr is None:
            self._jhr = np.zeros_like(self.gains)
        else:
            self._jhr.fill(0)
        return self._jhr

    def get_new_jhj(self):
        if self._jhj is None:
            self._jhj = np.zeros_like(self.gains)
            self._jhjinv = np.empty_like(self.gains)
        else:
            self._jhj.fill(0)
        return self._jhj, self._jhjinv

    def init_update(self, jhr):
        if self._update is None:
            self._update = np.zeros_like(jhr)
        else:
            self._update.fill(0)
        return self._update

    def get_obs_or_res(self, obser_arr, model_arr):
        if self.n_dir > 1:
            if self._r is None:
                self._r = np.empty_like(obser_arr)
            self.compute_residual(obser_arr, model_arr, self._r)
            return self._r
        else:
            return obser_arr

    def get_gain_mask(self):
        """Helper method: make a gain array mask from gain flags by broadcasting the corr1/2 axes."""
        mask = np.zeros_like(self.gains, bool)
        mask[:] = (self.gflags!=0)[...,np.newaxis,np.newaxis]
        return mask

    def init_gains(self):
        """
        Construct gain and flag arrays. Normally we have one gain/one flag per interval, but 
        subclasses may redefine this, if they deal with e.g. full-resolution gains.
        
        Sets up the following attributes: 
            gain_shape (list):
                shape of the gains array, i.e. (n_dir, NT, NF, n_ant, n_cor, n_cor)
                Default version has NT=n_timint, NF=n_freint, i.e. one gain per interval.
                 
            gain_grid (dict):
                grid on which gains are defined, as a dict of {'time': times, 'freq': frequencies},
                where times is a vector of NT points and frequencies is a vector of NF points.
                
            gains (np.ndarray):
                gains array of the specified shape
                
            gflags (np.ndarray):
                gain flags array of the same shape, minus the two correlation axes
                
            gain_intervals (list):
                intervals on which gains are defined. Default version has [t_int, f_int].
                
            _gainres_to_fullres (callable):
                function to go from an array of gain shape to full time/freq resolution
                (default uses unpack_intervals)
            
            _interval_to_gainres (callable):
                function to go from an array of interval shape to gain shape (default is identity)
        """
        # Construct the appropriate shape for the gains.
        self.gain_intervals = self.t_int, self.f_int
        self.gain_shape = [self.n_dir, self.n_timint, self.n_freint, self.n_ant, self.n_cor, self.n_cor]
        self.gain_grid = self.interval_grid

        self.gains = self.cykernel.allocate_gain_array(self.gain_shape, self.dtype)

        self.gains[:] = np.eye(self.n_cor)
        self.gflags = np.zeros(self.gain_shape[:-2], FL.dtype)

        # function used to unpack the gains or flags into full time/freq resolution
        self._gainres_to_fullres  = self.unpack_intervals
        # function used to unpack interval resolution to gain resolution
        self._interval_to_gainres = self.copy_or_identity

    def compute_residual(self, obser_arr, model_arr, resid_arr):
        """
        This function computes the residual. This is the difference between the
        observed data, and the model data with the gains applied to it.

        Args:
            obser_arr (np.ndarray): 
                Shape (n_mod, n_tim, n_fre, n_ant, n_ant, n_cor, n_cor) array containing the 
                observed visibilities.
            model_arr (np.ndrray): 
                Shape (n_dir, n_mod, n_tim, n_fre, n_ant, n_ant, n_cor, n_cor) array containing the 
                model visibilities.
            resid_arr (np.ndarray): 
                Shape (n_mod, n_tim, n_fre, n_ant, n_ant, n_cor, n_cor) array into which the 
                computed residuals should be placed.

        Returns:
            np.ndarray: 
                Array containing the result of computing D - GMG\ :sup:`H`.
        """

        gains_h = self.get_conj_gains()

        np.copyto(resid_arr, obser_arr)

        self.cykernel.cycompute_residual(model_arr, self.gains, gains_h, resid_arr, *self.gain_intervals)

        return resid_arr


    def apply_gains(self, model_arr):
        """
        Applies the gains to an array at full time-frequency resolution. 

        Args:
            model_arr (np.ndarray):
                Shape (n_dir, n_mod, n_tim, n_fre, n_ant, n_ant, n_cor, n_cor) array containing 
                model visibilities.

        Returns:
            np.ndarray:
                Array containing the result of GMG\ :sup:`H`.
        """
        gains_h = self.get_conj_gains()

        self.cykernel.cyapply_gains(model_arr, self.gains, gains_h, *self.gain_intervals)

        return model_arr

    def apply_inv_gains(self, obser_arr, corr_vis=None):
        """
        Applies the inverse of the gain estimates to the observed data matrix.

        Args:
            obser_arr (np.ndarray): 
                Shape (n_mod, n_tim, n_fre, n_ant, n_ant, n_cor, n_cor) array containing the 
                observed visibilities.
            corr_vis (np.ndarray or None, optional): 
                if specified, shape (n_mod, n_tim, n_fre, n_ant, n_ant, n_cor, n_cor) array 
                into which the corrected visibilities should be placed.

        Returns:
            np.ndarray: 
                Array containing the result of G\ :sup:`-1`\DG\ :sup:`-H`.
        """

        g_inv, gh_inv, flag_count = self.get_inverse_gains()

        if corr_vis is None:
            corr_vis = np.empty_like(obser_arr)

        self.cykernel.cycompute_corrected(obser_arr, g_inv, gh_inv, corr_vis, *self.gain_intervals)

        return corr_vis, flag_count

    @staticmethod
    def exportable_solutions():
        """ Returns a dictionary of exportable solutions for this machine type. """

        return { "gain": (1+0j, ("dir", "time", "freq", "ant", "corr1", "corr2")),
                 "gain.err": (0., ("dir", "time", "freq", "ant", "corr1", "corr2")) }

    def importable_solutions(self):
        """ Returns a dictionary of importable solutions for this machine type. """
        return { "gain": self.interval_grid }

    def export_solutions(self):
        """ Saves the solutions to a dict of {label: solutions,grids} items. """
        
        mask = self.get_gain_mask()

        sols = { "gain": (masked_array(self.gains, mask), self.gain_grid) }

        if self.posterior_gain_error is not None:
            sols["gain.err"] = (masked_array(self.posterior_gain_error, mask), self.gain_grid)
        else:
            sols["gain.err"] = (masked_array(np.zeros(self.gain_shape, float), np.ones(self.gain_shape, bool)), self.gain_grid)

        return sols

    def import_solutions(self, soldict):
        """ 
        Loads solutions from a dict. 
        
        Args:
            soldict (dict):
                Contains gains solutions which must be loaded.
        """

        sol = soldict.get("gain")
        if sol is not None:
            self.gains[:] = sol.data
            # collapse the corr1/2 axes
            self.gflags[sol.mask.any(axis=(-1,-2))] |= FL.MISSING
            self._gains_loaded = True
            self.restrict_solution()

    def precompute_attributes(self, data_arr, model_arr, flags_arr, inv_var_chan):
        """Precomputes various stats before starting a solution"""
        unflagged = MasterMachine.precompute_attributes(self, data_arr, model_arr, flags_arr, inv_var_chan)

        ## NB: not sure why I used to apply MISSING|PRIOR here. Surely other input flags must be honoured
        ## (SKIPSOL, NULLDATA, etc.)?
        ### Pre-flag gain solution intervals that are completely flagged in the input data
        ### (i.e. MISSING|PRIOR). This has shape (n_timint, n_freint, n_ant).

        missing_intervals = self.interval_and((flags_arr!=0).all(axis=-1))

        self.missing_gain_fraction = missing_intervals.sum() / float(missing_intervals.size)

        # convert the intervals array to gain shape, and apply flags
        self.gflags[:, self._interval_to_gainres(missing_intervals)] = FL.MISSING

        # number of data points per time/frequency/antenna
        numeq_tfa = unflagged.sum(axis=-1)

        # compute error estimates per direction, antenna, and interval
        with np.errstate(invalid='ignore', divide='ignore'):
            sigmasq = 1/inv_var_chan                        # squared noise per channel. Could be infinite if no data
            # collapse direction axis, if not directional
            if not self.dd_term:
                model_arr = model_arr.sum(axis=0, keepdims=True)
            # mean |model|^2 per direction+TFA
            modelsq = (model_arr*np.conj(model_arr)).real.sum(axis=(1,-1,-2,-3)) / \
                      (self.n_mod*self.n_cor*self.n_cor*numeq_tfa)
            modelsq[:, numeq_tfa==0] = 0
            # inverse SNR^2 per direction+TFA
            inv_snr2 = sigmasq[np.newaxis, np.newaxis, :, np.newaxis] / modelsq
            inv_snr2[:, numeq_tfa==0] = 0
            # take the mean SNR^-2 over each interval
            # numeq_tfa becomes number of points per interval, antenna
            numeq_tfa = self.interval_sum(numeq_tfa)
            inv_snr2_int = self.interval_sum(inv_snr2,1) / numeq_tfa[np.newaxis,...]
            inv_snr2_int[:, numeq_tfa==0] = 0
            # convert that into a gain error per direction,interval,antenna
            self.prior_gain_error = np.sqrt(inv_snr2_int /
                                      (self.eqs_per_interval - self.num_unknowns)[np.newaxis, :, :, np.newaxis])

        self.prior_gain_error[:, ~self.valid_intervals, :] = 0
        # reset to 0 for fixed directions
        if self.dd_term:
            self.prior_gain_error[self.fix_directions, ...] = 0

        # flag gains on max error
        bad_gain_intervals = self.prior_gain_error > self.max_gain_error    # dir,time,freq,ant
        if bad_gain_intervals.any():
            # (n_dir,) array showing how many were flagged per direction
            self._n_flagged_on_max_error = bad_gain_intervals.sum(axis=(1,2,3))
            # raised corresponding gain flags
            self.gflags[self._interval_to_gainres(bad_gain_intervals,1)] |= FL.LOWSNR
            self.prior_gain_error[bad_gain_intervals] = 0
            # flag intervals where all directions are bad, and propagate that out into flags
            bad_intervals = bad_gain_intervals.all(axis=0)
            if bad_intervals.any():
                bad_slots = self.unpack_intervals(bad_intervals)
                flags_arr[bad_slots,...] |= FL.LOWSNR
                unflagged[bad_slots,...] = False
                self._update_equation_counts(unflagged)
        else:
            self._n_flagged_on_max_error = None

        self._n_flagged_on_max_posterior_error = None
        self.flagged = self.gflags != 0
        self.n_flagged = self.flagged.sum()

        return unflagged
        # # get error estimate model
        # model = np.sqrt(self.interval_sum(modelsq,1))
        # self.model_error = model*self.gain_error*(2+self.gain_error)

    def _update_equation_counts(self, unflagged):
        """Sets up equation counters based on flagging information. Overrides base version to compute
        additional stuff"""
        MasterMachine._update_equation_counts(self, unflagged)

        self.eqs_per_interval = self.interval_sum(self.eqs_per_tf_slot)

        ndir = self.n_dir - len(self.fix_directions) if self.dd_term else 1
        self.num_unknowns = self.dof_per_antenna*self.n_ant*ndir

        # The following determines the number of valid (unflagged) time/frequency slots and the number
        # of valid solution intervals.

        self.valid_intervals = self.eqs_per_interval > self.num_unknowns
        self.num_valid_intervals = self.valid_intervals.sum()
        self.n_valid_sols = self.num_valid_intervals * self.n_dir

        if self.num_valid_intervals:
            # Adjust chi-sq normalisation based on DoF count: MasterMachine computes chi-sq normalization
            # as 1/N_eq, we want to compute it as the reduced chi-square statistic, 1/(N_eq-N_dof)
            # This results in a per-interval correction factor

            with np.errstate(invalid='ignore', divide='ignore'):
                corrfact = self.eqs_per_interval.astype(float)/(self.eqs_per_interval - self.num_unknowns)
            corrfact[~self.valid_intervals] = 0

            self._chisq_tf_norm_factor *= self.unpack_intervals(corrfact)
            self._chisq_norm_factor *= corrfact.sum() / self.num_valid_intervals


    def next_iteration(self):
        np.copyto(self.old_gains, self.gains)
        return MasterMachine.next_iteration(self)

    def flag_solutions(self, flags_arr, final):
        """ Flags gain solutions based on certain criteria, e.g. out-of-bounds, null, etc. """

        # Anything previously flagged for another reason will not be reflagged.

        flagged = self.flagged
        nfl0 = self.n_flagged

        # while iterating, flag on OOB and such
        if not final:
            gain_mags = np.abs(self.gains)

            # Check for inf/nan solutions. One bad correlation will trigger flagging for all correlations.

            boom = (~np.isfinite(self.gains)).any(axis=(-1,-2))
            self.gflags[boom&~flagged] |= FL.BOOM
            flagged |= boom
            gain_mags[boom] = 0

            # Check for gain solutions for which diagonal terms have gone to 0.

            gnull = (self.gains[..., 0, 0] == 0) | (self.gains[..., 1, 1] == 0)
            self.gflags[gnull&~flagged] |= FL.GNULL
            flagged |= gnull

            # Check for gain solutions which are out of bounds (based on clip thresholds).

            if self.clip_after<self.iters and self.clip_upper or self.clip_lower:
                goob = np.zeros(gain_mags.shape, bool)
                if self.clip_upper:
                    goob = gain_mags.max(axis=(-1, -2)) > self.clip_upper
                if self.clip_lower:
                    goob |= (gain_mags[...,0,0]<self.clip_lower) | (gain_mags[...,1,1,]<self.clip_lower)
                self.gflags[goob&~flagged] |= FL.GOOB
                flagged |= goob

        else:
            # else final flagging -- check the posterior error estimate
            if self.posterior_gain_error is not None:
                # reset to 0 for fixed directions
                if self.dd_term:
                    self.posterior_gain_error[self.fix_directions, ...] = 0
                # flag gains on max error
                # the last axis is correlation -- we flag if any correlation is flagged
                bad_gain_intervals = (self.posterior_gain_error > self.max_post_error).any(axis=(-1,-2))  # dir,time,freq,ant

                # mask high-variance gains that are not already otherwise flagged
                mask = self._interval_to_gainres(bad_gain_intervals, 1)&~flagged

                # raise FL.GVAR flag on these gains (and clear on all others!)
                self.gflags &= ~FL.GVAR
                self.gflags[mask] |= FL.GVAR
                flagged[mask] = True

                self._n_flagged_on_max_posterior_error = mask.sum(axis=(1, 2, 3)) if mask.any() else None

                # if bad_gain_intervals.any():
                #     # (n_dir,) array showing how many were flagged per direction
                #     self._n_flagged_on_max_error = bad_gain_intervals.sum(axis=(1, 2, 3))
                #     # raised corresponding gain flags
                #     mask = self._interval_to_gainres(bad_gain_intervals, 1)
                #     self.gflags[mask] |= FL.GVAR
                #     flagged[mask] = True
                # else:
                #     self._n_flagged_on_max_posterior_error = None

        # Count the gain flags, excluding those set a priori due to missing data.

        self.flagged = self.gflags != 0
        self.n_flagged = flagged.sum()

        # keep flagged gains at their previous values
        self.gains[self.flagged] = self.old_gains[self.flagged]

        if self.n_flagged > nfl0 and self.propagates_flags:
            # convert gain flags to full time/freq resolution, and add directions together
            nodir_flags = self._gainres_to_fullres(np.bitwise_or.reduce(self.gflags, axis=0))

            # We remove the FL.MISSING bit when propagating as this bit is pre-set for data flagged
            # as PRIOR|MISSING. This prevents every PRIOR but not MISSING flag from becoming MISSING.

            flags_arr |= nodir_flags[:,:,:,np.newaxis]&~FL.MISSING
            flags_arr |= nodir_flags[:,:,np.newaxis,:]&~FL.MISSING

            self._update_equation_counts(flags_arr==0)

            return True
        return False


    def num_gain_flags(self, mask=None):
        return int((self.gflags&(mask or ~FL.MISSING) != 0).sum()), self.gflags.size

    @property
    def dof_per_antenna(self):
        """This property returns the number of real degrees of freedom per antenna, per solution interval"""
        if self.update_type == "diag":
            dofs = 4
        elif self.update_type == "phase-diag":
            dofs = 2
        elif self.update_type == "amp-diag":
            dofs = 2
        else:
            dofs = 8
        return dofs

    @property
    def has_valid_solutions(self):
        return bool(self.n_valid_sols)

    @property
    def num_converged_solutions(self):
        return self.n_cnvgd

    def check_convergence(self, min_delta_g):
        """
        Updates the convergence parameters of the current time-frequency chunk. 

        Args:
            min_delta_g (float):
                Threshold for the minimum change in the gains - convergence criterion.
        """
        
        diff_g = np.square(np.abs(self.old_gains - self.gains))
        diff_g[self.flagged] = 0
        diff_g = diff_g.sum(axis=(-1,-2,-3))
        
        norm_g = np.square(np.abs(self.gains))
        norm_g[self.flagged] = 1
        norm_g = norm_g.sum(axis=(-1,-2,-3))

        norm_diff_g = diff_g/norm_g

        self.max_update = np.sqrt(np.max(diff_g))
        self.n_cnvgd = (norm_diff_g <= min_delta_g**2).sum()
        self._frac_cnvgd = self.n_cnvgd / float(norm_diff_g.size)

    def restrict_solution(self):
        """
        Restricts the solutions by, for example, selecting a reference antenna or taking only the 
        amplitude. 
        """
        # raise flag so updates of G^H and G^-1 are computed
        self._gh_update = self._ghinv_update = True

        if self.update_type == "diag":
            self.gains[...,(0,1),(1,0)] = 0
        elif self.update_type == "phase-diag":
            self.gains[...,(0,1),(1,0)] = 0
            gdiag = self.gains[...,(0,1),(0,1)]
            gnull = gdiag==0
            with np.errstate(invalid='ignore'):
                gdiag /= abs(gdiag)
            gdiag[gnull] = 0
        elif self.update_type == "amp-diag":
            self.gains[...,(0,1),(1,0)] = 0
            np.abs(self.gains, out=self.gains)
        for idir in self.fix_directions:
            self.gains[idir, ...] = self.old_gains[idir, ...]
            self.posterior_gain_error[idir, ...] = 0

    @staticmethod
    def copy_or_identity(array, time_ind=0, out=None):
        """Helper conversion method. Returns array itself, or copies it to out"""
        if out is None:
            return array
        np.copyto(out, array)
        return out

    def unpack_intervals(self, arr, tdim_ind=0, out=None):
        """
        Helper conversion method. Unpacks an array that has time/freq axes in terms of intervals into a shape
        that has time/freq axes at full resolution.

        Args:
            arr (np.ndarray):
                Array with adjacent time and frequency axes.
            tdim_ind (int, optional):
                Position of time axis in array axes. 

        Returns:
            np.ndarray:
                Array at full resolution.
        """
        if out is not None:
            out[:] = arr[tuple([slice(None)] * tdim_ind + [self.t_mapping, self.f_mapping])]
            return out
        else:
            return arr[tuple([slice(None)] * tdim_ind + [self.t_mapping, self.f_mapping])]

    def interval_sum(self, arr, tdim_ind=0, out=None):
        """
        Helper conversion method. Sums an array with full resolution time/freq axes into time/freq intervals.

        Args:
            arr (np.ndarray):
                Array with adjacent time and frequency axes.
            tdim_ind (int, optional):
                Position of time axis in array axes. 

        Returns:
            np.ndarray:
                Array with interval resolution.
        """

        return np.add.reduceat(np.add.reduceat(arr, self.t_bins, tdim_ind), self.f_bins, tdim_ind+1, out=out)

    def interval_and(self, arr, tdim_ind=0, out=None):
        """
        Helper method. Logical-ands an array with full resolution time/freq axes into time/freq 
        intervals.

        Args:
            arr (np.ndarray):
                Array with adjacent time and frequency axes.
            tdim_ind (int, optional):
                Position of time axis in array axes. 

        Returns:
            np.ndarray:
                Array with interval resolution.
        """

        return np.logical_and.reduceat(np.logical_and.reduceat(arr, self.t_bins, tdim_ind), 
                                                                    self.f_bins, tdim_ind+1, out=out)
    @property
    def converged_fraction(self):
        return self._frac_cnvgd

    @property
    def has_converged(self):
        """ Returns convergence status. """
        return self.converged_fraction >= self.min_quorum or self.iters >= self.maxiter

    @property
    def conditioning_status_string(self):
        """Returns conditioning status string"""
        if self.solvable:
            mineqs = self.eqs_per_interval[self.valid_intervals].min() if self.num_valid_intervals else 0
            maxeqs = self.eqs_per_interval.max()
            anteqs = (self.eqs_per_antenna!=0).sum()
            string = "{}: {}/{} ints".format(self.jones_label,
                                                self.num_valid_intervals, self.n_tf_ints)
            if self.num_valid_intervals:
                string += " ({}-{} EPI)".format(mineqs, maxeqs)
                if self.dd_term:
                    string += " {} dirs".format(self.n_dir)
                string += " {}/{} ants, MGE {}".format(anteqs, self.n_ant,
                    " ".join(["{:.3}".format(self.prior_gain_error[idir, :].max()) for idir in xrange(self.n_dir)]))
                if self._n_flagged_on_max_error is not None:
                    string += ", NFMGE {}".format(" ".join(map(str,self._n_flagged_on_max_error)))

            return string
        else:
            return "{}: n/s".format(self.jones_label)


    @property
    def current_convergence_status_string(self):
        """
        This property must return a status string for the gain machine, e.g.
            "G: 20 iters, conv 60.02%, g/fl 15.00%"
        """
        if self.solvable:
            string = "{}: {} iters, conv {:.2%}".format(self.jones_label, self.iters, self.converged_fraction)
            nfl, ntot = self.num_gain_flags()
            if nfl:
                string += ", g/fl {:.2%}".format(nfl/float(ntot))
            if self.missing_gain_fraction:
                string += ", d/fl {:.2%}".format(self.missing_gain_fraction)
            string += ", max update {:.4}".format(self.max_update)
            if self.posterior_gain_error is not None:
                string += ", PGE " + " ".join(["{:.3}".format(self.posterior_gain_error[idir, :].max())
                                               for idir in xrange(self.n_dir)])
            return string
        else:
            return "{}: n/s{}".format(self.jones_label, ", loaded" if self._gains_loaded else "")

    @property
    def final_convergence_status_string(self):
        """
        This property must return a status string for the gain machine, e.g.
            "G: 20 iters, conv 60.02%, g/fl 15.00%"
        """
        if self.solvable:
            string = "{}: {} iters, conv {:.2%}".format(self.jones_label, self.iters, self.converged_fraction)
            nfl, ntot = self.num_gain_flags()
            if nfl:
                string += ", g/fl {:.2%}".format(nfl/float(ntot))
            if self.missing_gain_fraction:
                string += ", d/fl {:.2%}".format(self.missing_gain_fraction)
            if self.posterior_gain_error is not None:
                string += ", PGE " + " ".join(["{:.3}".format(self.posterior_gain_error[idir, :].max())
                                               for idir in xrange(self.n_dir)])
            if self._n_flagged_on_max_posterior_error is not None:
                string += ", NFPGE {}".format(" ".join(map(str,self._n_flagged_on_max_posterior_error)))
            return string
        else:
            return "{}: n/s{}".format(self.jones_label, ", loaded" if self._gains_loaded else "")

    @property
    def has_stalled(self):
        """ Returns stalled status. """

        return self._has_stalled

    @has_stalled.setter
    def has_stalled(self, value):
        """ Sets stalled status. """

        self._has_stalled = value






