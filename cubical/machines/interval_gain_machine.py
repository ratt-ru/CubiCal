# CubiCal: a radio interferometric calibration suite
# (c) 2017 Rhodes University & Jonathan S. Kenyon
# http://github.com/ratt-ru/CubiCal
# This code is distributed under the terms of GPLv2, see LICENSE.md for details
from abc import ABCMeta, abstractmethod
import numpy as np
from cubical.flagging import FL
from cubical.machines.abstract_machine import MasterMachine
import cubical.kernels.cyfull_complex as cyfull

from numpy.ma import masked_array

class PerIntervalGains(MasterMachine):
    """
    This is a base class for all gain solution machines that use solutions intervals.
    """

    def __init__(self, label, data_arr, ndir, nmod, times, frequencies, options):
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

        MasterMachine.__init__(self, label, data_arr, ndir, nmod, times, frequencies, options)

        self.n_dir, self.n_mod = ndir, nmod
        _, self.n_tim, self.n_fre, self.n_ant, self.n_ant, self.n_cor, self.n_cor = data_arr.shape
    
        self.dtype = data_arr.dtype
        self.ftype = data_arr.real.dtype
        self.t_int = options["time-int"] or self.n_tim
        self.f_int = options["freq-int"] or self.n_fre
        self.eps = 1e-6

        # timestamp of start of each interval
        t0, f0 = times[0::self.t_int], frequencies[0::self.f_int]
        t1, f1 = t0.copy(), f0.copy()
        # timestamp of end of each interval -- need to take care if not evenly divisible
        t1[:-1] = times[self.t_int-1:-1:self.t_int]
        f1[:-1] = frequencies[self.f_int-1:-1:self.f_int]
        t1[-1], f1[-1] = times[-1], frequencies[-1]

        # interval_grid determines the per-interval grid poins
        self.interval_grid = dict(time=(t0+t1)/2, freq=(f0+f1)/2)
        # data_grid determines the full resolution grid
        self.data_grid = dict(time=times, freq=frequencies)

        # n_tim and n_fre are the time and frequency dimensions of the data arrays.
        # n_timint and n_freint are the time and frequnecy dimensions of the gains.

        self.n_timint = int(np.ceil(float(self.n_tim) / self.t_int))
        self.n_freint = int(np.ceil(float(self.n_fre) / self.f_int))
        self.n_tf_ints = self.n_timint * self.n_freint

        # Initialise attributes used for computing values over intervals.

        self.t_bins = range(0, self.n_tim, self.t_int)
        self.f_bins = range(0, self.n_fre, self.f_int)

        t_ind = np.arange(self.n_tim)//self.t_int
        f_ind = np.arange(self.n_fre)//self.f_int

        self.t_mapping, self.f_mapping = np.meshgrid(t_ind, f_ind, indexing="ij")

        # Initialise attributes used in convergence testing. n_cnvgd is the number
        # of solutions which have converged.

        self._has_stalled = False
        self.n_cnvgd = 0 
        self.iters = 0
        self.maxiter = options["max-iter"]
        self.min_quorum = options["conv-quorum"]
        self.update_type = options["update-type"]
        self.ref_ant = options["ref-ant"]
        self.dd_term = options["dd-term"]
        self.fix_directions = options["fix-dirs"] or []
        if type(self.fix_directions) is int:
            self.fix_directions = [self.fix_directions]
        # True if gains are loaded from a DB
        self._gains_loaded = False

        # Construct the appropriate shape for the gains.

        self.gain_shape = [self.n_dir, self.n_timint, self.n_freint, self.n_ant, self.n_cor, self.n_cor]
        self.gains = None

        # Construct flag array and populate flagging attributes.

        self.n_flagged = 0
        self.clip_lower = options["clip-low"]
        self.clip_upper = options["clip-high"]
        self.clip_after = options["clip-after"]
        self.flagbit = FL.ILLCOND

        self.init_gains()
        self.old_gains = self.gains.copy()

    def init_gains(self):
        """
        Construct gain and flag arrays. Normally we have one gain/one flag per interval, but 
        subclasses may redefine this, if they deal with full-resolution gains.
        """
        self.gain_grid = self.interval_grid
        self.gain_shape = [self.n_dir,self.n_timint,self.n_freint,self.n_ant,self.n_cor,self.n_cor]
        self.gains = np.empty(self.gain_shape, dtype=self.dtype)
        self.gains[:] = np.eye(self.n_cor)
        self.flag_shape = self.gain_shape[:-2]
        self.gflags = np.zeros(self.flag_shape,FL.dtype)
        # Total number of independent gain problems to be solved
        self.n_sols = float(self.n_dir * self.n_tf_ints)

        # function used to unpack the gains or flags into full time/freq resolution
        self._gainres_to_fullres  = self.unpack_intervals
        # function used to unpack interval resolution to gain resolution
        self._interval_to_gainres = lambda array,time_ind=0: array


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
        gh = self.gains.transpose(0,1,2,3,5,4).conj()

        cyfull.cyapply_gains(model_arr, self.gains, gh, self.t_int, self.f_int)

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

        g_inv = np.empty_like(self.gains)

        flag_count = cyfull.cycompute_jhjinv(self.gains, g_inv, self.gflags, self.eps,
                                             self.flagbit)  # Function can invert G.

        gh_inv = g_inv.transpose(0, 1, 2, 3, 5, 4).conj()

        if corr_vis is None:
            corr_vis = np.empty_like(obser_arr)

        cyfull.cycompute_corrected(obser_arr, g_inv, gh_inv, corr_vis, self.t_int, self.f_int)

        return corr_vis, flag_count

    @staticmethod
    def exportable_solutions():
        """ Returns a dictionary of exportable solutions for this machine type. """

        return { "gain": (1+0j, ("dir", "time", "freq", "ant", "corr1", "corr2")) }

    def importable_solutions(self):
        """ Returns a dictionary of importable solutions for this machine type. """
        return { "gain": self.interval_grid }

    def export_solutions(self):
        """ Saves the solutions to a dict of {label: solutions,grids} items. """
        
        # Make a mask from gain flags by broadcasting the corr1/2 axes.
        mask = np.zeros_like(self.gains, bool)
        mask[:] = (self.gflags!=0)[...,np.newaxis,np.newaxis]

        return { "gain".format(self.jones_label): (masked_array(self.gains, mask), self.gain_grid) }

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

    def update_stats(self, flags, eqs_per_tf_slot):
        """
        This method computes various stats and totals based on the current state of the flags.
        These values are used for weighting the chi-squared and doing intelligent convergence
        testing.

        Args:
            flags_arr (np.ndarray):
                Shape (n_tim, n_fre, n_ant, n_ant) array containing flags.
            eqs_per_tf_slot (np.ndarray):
                Shape (n_tim, n_fre) array containing a count of equations per time-frequency slot.
        """

        # (n_timint, n_freint) array containing number of valid equations per each time/freq interval.

        self.eqs_per_interval = self.interval_sum(eqs_per_tf_slot)

        # The following determines the number of valid (unflagged) time/frequency slots and the number
        # of valid solution intervals.

        self.valid_intervals = self.eqs_per_interval>0
        self.num_valid_intervals = self.valid_intervals.sum()

        # Pre-flag gain solution intervals that are completely flagged in the input data 
        # (i.e. MISSING|PRIOR). This has shape (n_timint, n_freint, n_ant).

        missing_intervals = self.interval_and((flags&(FL.MISSING|FL.PRIOR) != 0).all(axis=-1))

        self.missing_gain_fraction = missing_intervals.sum() / float(missing_intervals.size)

        # convert the intervals array to gain shape, and apply flags
        self.gflags[:, self._interval_to_gainres(missing_intervals)] = FL.MISSING

    def flag_solutions(self):
        """ Flags gain solutions based on certain criteria, e.g. out-of-bounds, null, etc. """
    
        gain_mags = np.abs(self.gains)

        # Anything previously flagged for another reason will not be reflagged.
        
        flagged = self.gflags != 0

        # Check for inf/nan solutions. One bad correlation will trigger flagging for all corrlations.

        boom = (~np.isfinite(self.gains)).any(axis=(-1,-2))
        self.gflags[boom&~flagged] |= FL.BOOM
        flagged |= boom

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

        # Count the gain flags, excluding those set a priori due to missing data.

        self.flagged = flagged
        self.n_flagged = (self.gflags&~FL.MISSING != 0).sum()

    def propagate_gflags(self, flags):
        """
        Propagates the flags raised by the gain machine back into the data. This is necessary as 
        the gain flags may not have the same shape as the data.

        Args:
            flags (np.ndarray):
                Shape (n_tim, n_fre, n_ant, n_ant) array containing flags. 
        """

        # convert gain flags to full time/freq resolution
        nodir_flags = self._gainres_to_fullres(np.bitwise_or.reduce(self.gflags, axis=0))

        flags |= nodir_flags[:,:,:,np.newaxis]&~FL.MISSING 
        flags |= nodir_flags[:,:,np.newaxis,:]&~FL.MISSING

    def update_conv_params(self, min_delta_g):
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

    def restrict_solution(self):
        """
        Restricts the solutions by, for example, selecting a reference antenna or taking only the 
        amplitude. 
        """

        if self.update_type == "diag":
            self.gains[...,(0,1),(1,0)] = 0
        elif self.update_type == "phase-diag":
            self.gains[...,(0,1),(1,0)] = 0
            self.gains[...,(0,1),(0,1)] = self.gains[...,(0,1),(0,1)]/np.abs(self.gains[...,(0,1),(0,1)])
        elif self.update_type == "amp-diag":
            self.gains[...,(0,1),(1,0)] = 0
            np.abs(self.gains, out=self.gains)
        for idir in self.fix_directions:
            self.gains[idir, ...] = self.old_gains[idir, ...]

    def update_term(self):
        """ Updates the current iteration. """

        self.iters += 1

    def unpack_intervals(self, arr, tdim_ind=0):
        """
        Helper method. Unpacks an array that has time/freq axes in terms of intervals into a shape
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

        return arr[[slice(None)] * tdim_ind + [self.t_mapping, self.f_mapping]]

    def interval_sum(self, arr, tdim_ind=0):
        """
        Helper method. Sums an array with full resolution time/freq axes into time/freq intervals.

        Args:
            arr (np.ndarray):
                Array with adjacent time and frequency axes.
            tdim_ind (int, optional):
                Position of time axis in array axes. 

        Returns:
            np.ndarray:
                Array with interval resolution.
        """

        return np.add.reduceat(np.add.reduceat(arr, self.t_bins, tdim_ind), self.f_bins, tdim_ind+1)

    def interval_and(self, arr, tdim_ind=0):
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
                                                                    self.f_bins, tdim_ind+1)

    @property
    def has_converged(self):
        """ Returns convergence status. """

        return self.n_cnvgd/self.n_sols > self.min_quorum or self.iters >= self.maxiter

    @property
    def status_string(self):
        """
        This property must return a status string for the gain machine, e.g.
            "G: 20 iters, conv 60.02%, g/fl 15.00%"
        """
        if self.solvable:
            return "{}: {} iters, conv {:.2%}{}{}".format(
                self.jones_label, self.iters,
                self.n_cnvgd/self.n_sols,
                ", g/fl {:.2%}".format(self.n_flagged/self.gflags.size) if self.n_flagged else "",
                ", d/fl {:.2%}".format(self.missing_gain_fraction) if self.missing_gain_fraction else "")
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






