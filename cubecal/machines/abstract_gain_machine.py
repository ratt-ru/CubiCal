from abc import ABCMeta, abstractmethod
import math

class PerIntervalGains(object):
    """
    This is a base class for all gain solution machines that use solutions intervals.
    """

    __metaclass__ = ABCMeta

    def __init__(self, model_arr, options):
        """
        Given a model array, initializes various sizes relevant to the gain solutions.
        """

        self.n_dir, self.n_mod, self.n_tim, self.n_fre, self.n_ant, self.n_ant, self.n_cor, self.n_cor = model_arr.shape
        self.dtype = model_arr.dtype
        self.t_int = options["time-int"]
        self.f_int = options["freq-int"]
        self.eps = 1e-6

        # n_tim and n_fre are the time and frequency dimensions of the data arrays.
        # n_timint and n_freint are the time and frequnecy dimensions of the gains.

        self.n_timint = int(math.ceil(float(self.n_tim) / self.t_int))  # Number of time intervals
        self.n_freint = int(math.ceil(float(self.n_fre) / self.f_int))  # Number of freq intervals
        self.n_tf = self.n_fre * self.n_tim         # Number of time-freq slots
        self.n_int = self.n_timint * self.n_freint  # Total number of solution intervals

        # Total number of solutions.

        self.n_sols = float(self.n_dir * self.n_int)        

        # Construct the appropriate shape for the gains.

        self.gain_shape = [self.n_dir, self.n_timint, self.n_freint, self.n_ant, self.n_cor, self.n_cor]
        self.flag_shape = [self.n_timint, self.n_freint, self.n_ant]

    @abstractmethod
    def compute_js(self):
        return NotImplementedError

    @abstractmethod
    def compute_update(self):
        return NotImplementedError

    @abstractmethod
    def compute_residual(self):
        return NotImplementedError

    @abstractmethod
    def apply_inv_gains(self):
        return NotImplementedError

    def precompute_attributes(self, *args, **kwargs):
        return
