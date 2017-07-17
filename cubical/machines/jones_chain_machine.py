from cubical.machines.abstract_machine import MasterMachine
from cubical.machines.complex_2x2_machine import Complex2x2Gains
import numpy as np
import cubical.kernels.cyfull_complex as cyfull
import cubical.kernels.cychain as cychain

class JonesChain(MasterMachine):
    """
    This class implements a gain machine for an arbitrary chain of Jones matrices.
    """
    def __init__(self, model_arr, times, frequencies, options):
        
        MasterMachine.__init__(self, times, frequencies)

        self.n_dir, self.n_mod, self.n_tim, self.n_fre, self.n_ant, self.n_ant, self.n_cor, self.n_cor = model_arr.shape

        # This instantiates the number of complex 2x2 elements in our chain. Each element is a 
        # gain machine in its own right - the purpose of this machine is to manage these machines
        # and do the relevant fiddling between parameter updates. 
        # TODO: Figure out parameter specification for this.

        self.n_terms = options["sol"]["n-terms"]

        self.jones_terms = [Complex2x2Gains(model_arr, times, frequencies, options["j{}".format(n)]) 
                                                                 for n in xrange(1, self.n_terms+1)]
        
        self.active_index = 0       

    def compute_js(self, obser_arr, model_arr):
        """
        This method is expected to compute (J^HJ)^-1 and J^HR. In practice, this method can be 
        very flexible, as it is only used in the compute_update method and need only be consistent
        with that usage. Should support the use of both the true residual and the observed data. 
        """     

        # NB: This may be misleading - we are technically getting the n_tint and n_fint sizes here.

        n_dir, n_tint, n_fint, n_ant, n_cor, n_cor = self.gains.shape

        current_model_arr = model_arr.copy()

        for ind in xrange(self.n_terms - 1, self.active_index, -1):
            term = self.jones_terms[ind]
            term.apply_gains(current_model_arr)

        jh = np.zeros_like(current_model_arr)

        for ind in xrange(self.active_index, -1, -1):
            term = self.jones_terms[ind]
            cyfull.cycompute_jh(current_model_arr, term.gains, jh, term.t_int, term.f_int)
            current_model_arr[:] = jh 
        
        jhr_shape = [n_dir, self.n_tim, self.n_fre, n_ant, n_cor, n_cor]

        jhr = np.zeros(jhr_shape, dtype=obser_arr.dtype)

        # TODO: This breaks with the new compute residual code for n_dir > 1. Will need a fix.
        if n_dir > 1:
            resid_arr = np.empty_like(obser_arr)
            r = self.compute_residual(obser_arr, current_model_arr, resid_arr)
        else:
            r = obser_arr

        # NEED TO STOP AND THINK HERE. IS IT WORTH WRITING A FULL SET OF CHAIN KERNELS? THE CURRENT
        # APPROACH IS GETTING HACKY. IN PRINCIPLE, I NEED TO DEFER SUMMATION OVER SOLUTION INTERVALS
        # TILL AFTER I HAVE APPLIED THE LEFT HAND GAINS.

        cyfull.cycompute_jhr(jh, r, jhr, 1, 1)

        for ind in xrange(0, self.active_index, 1):
            term = self.jones_terms[ind]
            g_inv = np.empty_like(term.gains)
            cyfull.cycompute_jhjinv(term.gains, g_inv, term.gflags, term.eps, term.flagbit)
            cychain.cyapply_left_inv_jones(jhr, g_inv, term.t_int, term.f_int)

        jhrint_shape = [n_dir, n_tint, n_fint, n_ant, n_cor, n_cor]
        
        jhrint = np.zeros(jhrint_shape, dtype=jhr.dtype)

        cychain.cysum_jhr_intervals(jhr, jhrint, self.t_int, self.f_int)

        jhj = np.zeros(jhrint_shape, dtype=obser_arr.dtype)

        cyfull.cycompute_jhj(jh, jhj, self.t_int, self.f_int)

        jhjinv = np.empty(jhrint_shape, dtype=obser_arr.dtype)

        flag_count = cyfull.cycompute_jhjinv(jhj, jhjinv, self.gflags, self.eps, self.flagbit)

        return jhrint, jhjinv, flag_count

    def compute_update(self, model_arr, obser_arr):
        """
        This method is expected to compute the parameter update. As such, it must fetch or compute 
        the terms of the update in order to update the gains. Should call the compute_js but is 
        very flexible, provided it ultimately updates the gains. 
        """

        # This function shouldn't mimic the underlying machine - the update step is fundamentally 
        # different for the chain case. We need to take into account both a pre-compute and 
        # post-compute step BEFORE updating the gains.

        # This is currently a bit dodgy - convergence is checked for the previous active term. Might
        # be necessary to let the solver know about this bit of the chain. Alternatively, the solver
        # level while loop should become an infinite loop which instead queries the active term to 
        # determine whehther it has converged. I also need to deal with having one Jones term con-
        # verge before the others. This does make querying convergence on the underlying machines 
        # seem like a good choice. In this way, if a machine claims it is done, we can skip to the
        # next element in the chain. This will also give us the flexibility to ask for a single term
        # to be solved to completion before moving on to the next term in the chain. 

        jhr, jhjinv, flag_count = self.compute_js(obser_arr, model_arr)

        update = np.empty_like(jhr)

        cyfull.cycompute_update(jhr, jhjinv, update)

        if model_arr.shape[0]>1:
            update = self.gains + update

        if self.iters % 2 == 0:
            self.gains = 0.5*(self.gains + update)
        else:
            self.gains = update

        return flag_count

    def update_term(self):
        """
        This function will determine which element in the Jones chain is active. This can be 
        expanded to support complex convergence functionality.
        """

        if (self.iters)%2==0:
            self.active_index = (self.active_index + 1)%self.n_terms

    def compute_residual(self, obser_arr, model_arr, resid_arr):
        """
        This function computes the residual. This is the difference between the
        observed data, and the model data with the gains applied to it.

        Args:
            resid_arr (np.array): Array which will receive residuals.
                              Shape is n_dir, n_tim, n_fre, n_ant, a_ant, n_cor, n_cor
            obser_arr (np.array): Array containing the observed visibilities.
                              Same shape
            model_arr (np.array): Array containing the model visibilities.
                              Same shape
            gains (np.array): Array containing the current gain estimates.
                              Shape of n_dir, n_timint, n_freint, n_ant, n_cor, n_cor
                              Where n_timint = ceil(n_tim/t_int), n_fre = ceil(n_fre/t_int)

        Returns:
            residual (np.array): Array containing the result of computing D-GMG^H.
        """

        current_model_arr = model_arr.copy()

        for ind in xrange(self.n_terms-1, -1, -1): 
            term = self.jones_terms[ind]
            term.apply_gains(current_model_arr)

        resid_arr[:] = obser_arr

        cychain.cycompute_residual(current_model_arr, resid_arr)

        return resid_arr

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
            self.active_term.update_stats(flags, eqs_per_tf_slot)
        else:
            [term.update_stats(flags, eqs_per_tf_slot) for term in self.jones_terms]
   
    def update_conv_params(self, min_delta_g):
        """
        This method should check the convergence of the current time-frequency chunk. Should return 
        a Boolean.
        """

        self.active_term.update_conv_params(min_delta_g)

    def flag_solutions(self):
        """
        This method should do solution flagging based on the gains.
        """

        self.active_term.flag_solutions()

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

    @gains.setter
    def gains(self, value):
        self.active_term.gains = value

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
        self.active_term.old_gains = value

    @property
    def dtype(self):
        return self.active_term.dtype

    @property
    def ftype(self):
        return self.active_term.ftype

    @property
    def active_term(self):
        return self.jones_terms[self.active_index]

    @property
    def t_int(self):
        return self.active_term.t_int

    @property
    def f_int(self):
        return self.active_term.f_int

    @property
    def eps(self):
        return self.active_term.eps

    @property
    def flagbit(self):
        return self.active_term.flagbit

    @property
    def iters(self):
        return self.active_term.iters

    @iters.setter
    def iters(self, value):
        self.active_term.iters = value

    @property
    def maxiter(self):
        return self.active_term.maxiter

    @property
    def min_quorum(self):
        return self.active_term.min_quorum

    @property
    def has_converged(self):
        return self.active_term.has_converged

    @property
    def has_stalled(self):
        return self.active_term.has_stalled

    @has_stalled.setter   
    def has_stalled(self, value):
        self.active_term.has_stalled = value