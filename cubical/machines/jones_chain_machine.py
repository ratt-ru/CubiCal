from cubical.machines.abstract_machine import MasterMachine
from cubical.machines.complex_2x2_machine import Complex2x2Gains
import numpy as np
import cubical.kernels.cyfull_complex as cyfull
import sys

class JonesChain(MasterMachine):
    """
    This class implements a gain machine for an arbitrary chain of Jones matrices.
    """
    def __init__(self, model_arr, times, frequencies, options):
        
        MasterMachine.__init__(self, times, frequencies)

        self.n_dir, self.n_mod, self.n_tim, self.n_fre, self.n_ant, self.n_ant, self.n_cor, self.n_cor = model_arr.shape
        
        self.dtype = model_arr.dtype
        self.ftype = model_arr.real.dtype

        # This instantiates the number of complex 2x2 elements in our chain. Each element is a 
        # gain machine in its own right - the purpose of this machine is to manage these machines
        # and do the relevant fiddling between parameter updates. 
        # TODO: Figure out parameter specification for this.

        self.jones_terms = [Complex2x2Gains(model_arr, times, frequencies, options) for i in xrange(3)]
        self.active_term = self.jones_terms[0]

    def compute_js(self, obser_arr, model_arr):
        """
        This method is expected to compute (J^HJ)^-1 and J^HR. In practice, this method can be 
        very flexible, as it is only used in the compute_update method and need only be consistent
        with that usage. Should support the use of both the true residual and the observed data. 
        """
        
        self.active_term.compute_js(obser_arr, model_arr)

    def compute_update(self, model_arr, obser_arr, iters):
        """
        This method is expected to compute the parameter update. As such, it must fetch or compute 
        the terms of the update in order to update the gains. Should call the compute_js but is 
        very flexible, provided it ultimately updates the gains. 
        """
        
        self.active_term.compute_update(model_arr, obser_arr, iters)

    def compute_residual(self, obser_arr, model_arr, resid_arr):
        """
        This method should compute the residual at the the full time-frequency resolution of the
        data. Should return the residual.
        """
        
        self.active_term.compute_residual(obser_arr, model_arr, resid_arr)

    def apply_inv_gains(self, obser_arr, corr_vis=None):
        """
        This method should be able to apply the inverse of the gains to an array at full time-
        frequency resolution. Should return the input array at full resolution after the application
        of the inverse gains.
        """

        self.active_term.apply_inv_gains(obser_arr, corr_vis)

    def apply_gains(self):
        """
        This method should be able to apply the gains to an array at full time-frequency
        resolution. Should return the input array at full resolution after the application of the 
        gains.
        """
        return
           
    def update_stats(self, flags, eqs_per_tf_slot):
        """
        This method should compute a variety of useful parameters regarding the conditioning and 
        degrees of freedom of the current time-frequency chunk. Specifically, it must populate 
        an attribute containing the degrees of freedom per time-frequency slot. 
        """

        if hasattr(self.active_term, 'num_valid_intervals'):
            self.active_term.update_stats()
        else:
            [term.update_stats(flags, eqs_per_tf_slot) for term in self.jones_terms]
   
    def update_conv_params(self, min_delta_g):
        """
        This method should check the convergence of the current time-frequency chunk. Should return 
        a Boolean.
        """

        self.active_term.update_conv_params(min_delta_g)

    def flag_solutions(self, clip_gains=False):
        """
        This method should do solution flagging based on the gains.
        """

        self.active_term.flag_solutions(clip_gains)

    def propagate_gflags(self, flags):
        """
        This method should propagate the flags raised by the gain machine back into the data.
        This is necessary as the gain flags may not have the same shape as the data.
        """
        
        self.active_term.propagate_gflags(flags)

    def attr_print(self, attr_name):

        print self.attr_list(attr_name)

    def attr_list(self, attr_name):

        return [getattr(term, attr_name, None) for term in self.jones_terms]

    @property
    def gains(self):
        return self.active_term.gains

    @property
    def gflags(self):
        return self.active_term.gflags

    @property
    def n_cnvgd(self):
        return self.active_term.n_cnvgd

    @property
    def n_sols(self):
        return self.active_term.n_sols

    @property
    def eqs_per_interval(self):
        return self.active_term.eqs_per_interval

    @property
    def valid_intervals(self):
        return self.active_term.valid_intervals

    @property
    def n_tf_ints(self):
        return self.active_term.n_tf_ints

    @property
    def max_update(self):
        return self.active_term.max_update

    @property
    def n_flagged(self):
        return self.active_term.n_flagged

    @property
    def num_valid_intervals(self):
        return self.active_term.num_valid_intervals

    @property
    def missing_gain_fraction(self):
        return self.active_term.missing_gain_fraction

    @property
    def old_gains(self):
        return self.active_term.old_gains

    @old_gains.setter
    def old_gains(self, value):
        self.active_term.old_gains = self.active_term.gains