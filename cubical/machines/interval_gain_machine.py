from abc import ABCMeta, abstractmethod
import numpy as np
from cubical.flagging import FL
from cubical.machines.abstract_machine import MasterMachine

class PerIntervalGains(MasterMachine):
    """
    This is a base class for all gain solution machines that use solutions intervals.
    """

    def __init__(self, model_arr, times, frequencies, options):
        """
        Given a model array, initializes various sizes relevant to the gain solutions.
        """

        MasterMachine.__init__(self, times, frequencies)

        self.n_dir, self.n_mod, self.n_tim, self.n_fre, self.n_ant, self.n_ant, self.n_cor, self.n_cor = model_arr.shape
    
        self.dtype = model_arr.dtype
        self.ftype = model_arr.real.dtype
        self.t_int = options["time-int"]
        self.f_int = options["freq-int"]
        self.eps = 1e-6

        # n_tim and n_fre are the time and frequency dimensions of the data arrays.
        # n_timint and n_freint are the time and frequnecy dimensions of the gains.

        self.n_tf_slots = self.n_fre * self.n_tim
        self.n_timint = int(np.ceil(float(self.n_tim) / self.t_int))
        self.n_freint = int(np.ceil(float(self.n_fre) / self.f_int))
        self.n_tf_ints = self.n_timint * self.n_freint

        # Total number of solutions.

        self.n_sols = float(self.n_dir * self.n_tf_ints)

        # Initialise attributes used in convergence testing. n_cnvgd is the number
        # of solutions which have converged, n_stall is the number of solutions 
        # that have stalled chi-squared values and n_vis2x2 is the number of 
        # two-by-two visibility blocks. 

        self.n_cnvgd = 0 
        self.n_stall = 0
        self.n_2x2vis = self.n_tf_slots * self.n_ant * self.n_ant

        # Construct the appropriate shape for the gains.

        self.gain_shape = [self.n_dir, self.n_timint, self.n_freint, self.n_ant, self.n_cor, self.n_cor]

        # Construct flag array

        self.flag_shape = [self.n_dir, self.n_timint, self.n_freint, self.n_ant]
        self.gflags = np.zeros(self.flag_shape, FL.dtype)
        self.flagbit = FL.ILLCOND

    def compute_stats(self, flags):
        """
        This method computes various stats and totals based on the current state of the flags.
        These values are used for weighting the chi-squared and doing intelligent convergence
        testing.
        """

        unflagged = (flags==0)

        # Compute number of terms in each chi-square sum. Shape is (n_tim, n_fre, n_ant).

        self.nterms = 2 * self.n_cor * self.n_cor * np.sum(unflagged, axis=3)

        # (n_ant) vector containing the number of valid equations per antenna.
        # Factor of two is necessary as we have the conjugate of each equation too.

        self.eqs_per_antenna = 2*np.sum(unflagged, axis=(0, 1, 2)) * self.n_mod

        # (n_tim, n_fre) array containing number of valid equations for each time/freq slot.
        
        eqs_per_tf_slot = np.sum(unflagged, axis=(-1, -2)) * self.n_mod * self.n_cor * self.n_cor * 2

        # (n_timint, n_freint) array containing number of valid equations per each time/freq interval.

        self.eqs_per_interval = self.interval_sum(eqs_per_tf_slot)

        # The following determines the number of valid (unflagged) time/frequency slots and the number
        # of valid solution intervals.

        self.valid_intervals = self.eqs_per_interval>0
        self.num_valid_intervals = self.valid_intervals.sum()

        # Compute chi-squared normalization factor for each solution interval (used by compute_chisq() below)

        self.chisq_norm = np.zeros_like(self.eqs_per_interval, dtype=self.ftype)
        self.chisq_norm[self.valid_intervals] = (1. / self.eqs_per_interval[self.valid_intervals])



    def interval_sum(self, inarray, tdim_ind=0):

        old_shape = list(inarray.shape)
        n_dim = len(old_shape)

        n_lead = tdim_ind
        n_trail = n_dim - (n_lead + 2)

        t_dim = old_shape[tdim_ind]
        f_dim = old_shape[tdim_ind+1]

        t_bins = range(0, t_dim, self.t_int) if t_dim%self.t_int==0 else range(0, t_dim, self.t_int) + [t_dim]
        f_bins = range(0, f_dim, self.t_int) if f_dim%self.f_int==0 else range(0, f_dim, self.f_int) + [f_dim]

        t_slices = [slice(t0, t1) for (t0,t1) in zip(t_bins[:-1], t_bins[1:])]
        f_slices = [slice(f0, f1) for (f0,f1) in zip(f_bins[:-1], f_bins[1:])]

        makeslice = lambda lead, trail, t_sl, f_sl: [slice(None)]*lead + [t_sl, f_sl] + [slice(None)]*trail 

        tf_slices = [makeslice(n_lead, n_trail, t, f) for t in t_slices for f in f_slices]

        new_shape = old_shape[:tdim_ind] + [self.n_timint, self.n_freint] + old_shape[tdim_ind + 2:]

        outarray = np.array([np.sum(inarray[s], axis=(tdim_ind, tdim_ind+1)) for s in tf_slices]).reshape(new_shape)

        return outarray

