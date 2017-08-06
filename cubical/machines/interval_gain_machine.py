from abc import ABCMeta, abstractmethod
import numpy as np
from cubical.flagging import FL
from cubical.machines.abstract_machine import MasterMachine
from functools import partial

class PerIntervalGains(MasterMachine):
    """
    This is a base class for all gain solution machines that use solutions intervals.
    """

    def __init__(self, label, data_arr, ndir, nmod, times, frequencies, options):
        """
        Given a model array, initializes various sizes relevant to the gain solutions.
        """

        MasterMachine.__init__(self, label, times, frequencies, options)

        self.n_dir, self.n_mod = ndir, nmod
        _, self.n_tim, self.n_fre, self.n_ant, self.n_ant, self.n_cor, self.n_cor = data_arr.shape
    
        self.dtype = data_arr.dtype
        self.ftype = data_arr.real.dtype
        self.t_int = options["time-int"]
        self.f_int = options["freq-int"]
        self.eps = 1e-6

        # timestamp of start of each interval
        t0, f0 = times[0::self.t_int], frequencies[0::self.f_int]
        t1, f1 = t0.copy(), f0.copy()
        # timestamp of end of each interval -- need to take care if not evenly divisible
        t1[:-1] = times[self.t_int-1:-1:self.t_int]
        f1[:-1] = frequencies[self.f_int-1:-1:self.f_int]
        t1[-1], f1[-1] = times[-1], frequencies[-1]
        self._grid = dict(time=(t0+t1)/2, freq=(f0+f1)/2)

        # n_tim and n_fre are the time and frequency dimensions of the data arrays.
        # n_timint and n_freint are the time and frequnecy dimensions of the gains.

        self.n_timint = int(np.ceil(float(self.n_tim) / self.t_int))
        self.n_freint = int(np.ceil(float(self.n_fre) / self.f_int))
        self.n_tf_ints = self.n_timint * self.n_freint

        # Total number of solutions.

        self.n_sols = float(self.n_dir * self.n_tf_ints)

        # Initialise attributes used for computing values over intervals.

        self.t_bins = range(0, self.n_tim, self.t_int)
        self.f_bins = range(0, self.n_fre, self.f_int)

        t_ind = np.arange(self.n_tim)//self.t_int
        f_ind = np.arange(self.n_fre)//self.f_int

        self.t_mapping, self.f_mapping = np.meshgrid(t_ind, f_ind, indexing="ij")

        # Initialise attributes used in convergence testing. n_cnvgd is the number
        # of solutions which have converged.

        self.n_cnvgd = 0 

        # Construct the appropriate shape for the gains.

        self.gain_shape = [self.n_dir, self.n_timint, self.n_freint, self.n_ant, self.n_cor, self.n_cor]
        self.gains = None

        # Construct flag array and populate flagging attributes.

        self.n_flagged = 0
        self.clip_lower = options["clip-low"]
        self.clip_upper = options["clip-high"]
        self.flag_shape = [self.n_dir, self.n_timint, self.n_freint, self.n_ant]
        self.gflags = np.zeros(self.flag_shape, FL.dtype)
        self.flagbit = FL.ILLCOND

    # describe our solutions
    exportable_solutions = { "gain": (complex, ("dir", "time", "freq", "ant", "corr1", "corr2")) }
    importable_solutions = [ "gain" ]

    def get_solutions_grid(self):
        return self._grid

    def export_solutions(self):
        """This method saves the solutions to a dict"""
        return dict(gain=self.gains)

    def import_solutions(self, soldict):
        """This method loads solutions from an array"""
        self.gains[:] = soldict["gain"]

    def update_stats(self, flags, eqs_per_tf_slot):
        """
        This method computes various stats and totals based on the current state of the flags.
        These values are used for weighting the chi-squared and doing intelligent convergence
        testing.
        """

        # (n_timint, n_freint) array containing number of valid equations per each time/freq interval.

        self.eqs_per_interval = self.interval_sum(eqs_per_tf_slot)

        # The following determines the number of valid (unflagged) time/frequency slots and the number
        # of valid solution intervals.

        self.valid_intervals = self.eqs_per_interval>0
        self.num_valid_intervals = self.valid_intervals.sum()

        # Pre-flag gain solution intervals that are completely flagged in the input data 
        # (i.e. MISSING|PRIOR). This has shape (n_timint, n_freint, n_ant).

        missing_gains = self.interval_and((flags&(FL.MISSING|FL.PRIOR) != 0).all(axis=-1))

        # Gain flags have shape (n_dir, n_timint, n_freint, n_ant). All intervals with no prior data
        # are flagged as FL.MISSING.
        
        self.gflags[:, missing_gains] = FL.MISSING
        self.missing_gain_fraction = missing_gains.sum() / float(missing_gains.size)

    def flag_solutions(self, clip_gains=False):
        """
        This method will do basic flagging of the gain solutions.
        """

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

        if clip_gains and self.clip_upper or self.clip_lower:
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

        nodir_flags = self.unpack_intervals(np.bitwise_or.reduce(self.gflags, axis=0))

        flags |= nodir_flags[:,:,:,np.newaxis]&~FL.MISSING 
        flags |= nodir_flags[:,:,np.newaxis,:]&~FL.MISSING

    def update_conv_params(self, min_delta_g):
        
        diff_g = np.square(np.abs(self.old_gains - self.gains))
        diff_g[self.flagged] = 0
        diff_g = diff_g.sum(axis=(-1,-2,-3))
        
        norm_g = np.square(np.abs(self.gains))
        norm_g[self.flagged] = 1
        norm_g = norm_g.sum(axis=(-1,-2,-3))

        norm_diff_g = diff_g/norm_g

        self.max_update = np.max(diff_g)
        self.n_cnvgd = (norm_diff_g <= min_delta_g**2).sum()

    def unpack_intervals(self, arr, tdim_ind=0):

        return arr[[slice(None)] * tdim_ind + [self.t_mapping, self.f_mapping]]

    def interval_sum(self, arr, tdim_ind=0):
   
        return np.add.reduceat(np.add.reduceat(arr, self.t_bins, tdim_ind), self.f_bins, tdim_ind+1)

    def interval_and(self, arr, tdim_ind=0):
   
        return np.logical_and.reduceat(np.logical_and.reduceat(arr, self.t_bins, tdim_ind), self.f_bins, tdim_ind+1)






