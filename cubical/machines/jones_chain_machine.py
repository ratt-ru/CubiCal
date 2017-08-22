from cubical.machines.abstract_machine import MasterMachine
from cubical.machines.complex_2x2_machine import Complex2x2Gains
import numpy as np
import cubical.kernels.cyfull_complex as cyfull
import cubical.kernels.cychain as cychain

class JonesChain(MasterMachine):
    """
    This class implements a gain machine for an arbitrary chain of Jones matrices.
    """
    def __init__(self, label, data_arr, ndir, nmod, times, frequencies, jones_options):
        
        MasterMachine.__init__(self, label, data_arr, ndir, nmod, times, frequencies, jones_options)

        self.n_dir, self.n_mod = ndir, nmod
        _, self.n_tim, self.n_fre, self.n_ant, self.n_ant, self.n_cor, self.n_cor = data_arr.shape

        # This instantiates the number of complex 2x2 elements in our chain. Each element is a 
        # gain machine in its own right - the purpose of this machine is to manage these machines
        # and do the relevant fiddling between parameter updates. When combining DD terms with
        # DI terms, we need to be initialise the DI terms using only one direction - we do this with 
        # slicing rather than summation as it is slightly faster. 

        self.n_terms = len(jones_options)

        self.jones_terms = []
        for term_opts in jones_options:
            self.jones_terms.append(Complex2x2Gains(term_opts["label"], data_arr, ndir if term_opts["dd-term"] else 1,
                                                    nmod, times, frequencies, term_opts))

        self.active_index = 0

    def export_solutions(self):
        """This method saves the solutions to a dict of {label: solutions,grids} items"""
        soldict = {}
        for term in self.jones_terms:
            soldict.update(term.export_solutions())
        return soldict

    def importable_solutions(self):
        """This method loads solutions from a dict"""
        soldict = {}
        for term in self.jones_terms:
            soldict.update(term.importable_solutions())
        return soldict

    def import_solutions(self, soldict):
        """This method loads solutions from a dict"""
        for term in self.jones_terms:
            term.import_solutions(soldict)

    def compute_js(self, obser_arr, model_arr):
        """
        This method is expected to compute (J^HJ)^-1 and J^HR. In practice, this method can be 
        very flexible, as it is only used in the compute_update method and need only be consistent
        with that usage. Should support the use of both the true residual and the observed data. 
        """     

        n_dir, n_tint, n_fint, n_ant, n_cor, n_cor = self.gains.shape

        current_model_arr = model_arr.copy()

        for ind in xrange(self.n_terms - 1, self.active_index, -1):
            term = self.jones_terms[ind]
            term.apply_gains(current_model_arr)

        if not self.dd_term and self.n_dir>1:
            current_model_arr = np.sum(current_model_arr, axis=0, keepdims=True)

        jh = np.zeros_like(current_model_arr)

        for ind in xrange(self.active_index, -1, -1):
            term = self.jones_terms[ind]
            cyfull.cycompute_jh(current_model_arr, term.gains, jh, term.t_int, term.f_int)
            # print np.allclose(self.alternate_jh(current_model_arr, term.gains, jh, term.t_int, term.f_int),jh)
            current_model_arr[:] = jh 

        jhr_shape = [n_dir if self.dd_term else 1, self.n_tim, self.n_fre, n_ant, n_cor, n_cor]

        jhr = np.zeros(jhr_shape, dtype=obser_arr.dtype)

        if n_dir > 1:
            resid_arr = np.empty_like(obser_arr)
            r = self.compute_residual(obser_arr, model_arr, resid_arr)
        else:
            r = obser_arr

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

        # if not(self.dd_term) and model_arr.shape[0]>1:
        #     jhr, jhjinv, flag_count = self.compute_js(obser_arr, np.sum(model_arr, axis=0, keepdims=True))
        # else:
        #     jhr, jhjinv, flag_count = self.compute_js(obser_arr, model_arr)

        jhr, jhjinv, flag_count = self.compute_js(obser_arr, model_arr)

        update = np.empty_like(jhr)

        cyfull.cycompute_update(jhr, jhjinv, update)

        if self.dd_term and model_arr.shape[0]>1:
            update = self.gains + update

        if self.iters % 2 == 0 or self.dd_term:
            self.gains = 0.5*(self.gains + update)
        else:
            self.gains = update

        if self.update_type == "diag":
            self.gains[...,(0,1),(1,0)] = 0
        elif self.update_type == "phase-diag":
            self.gains[...,(0,1),(1,0)] = 0
            self.gains[...,(0,1),(0,1)] = self.gains[...,(0,1),(0,1)]/np.abs(self.gains[...,(0,1),(0,1)])

        # print self.gains[:,0,0,8,:]

        return flag_count

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

    def apply_inv_gains(self, resid_vis, corr_vis=None):
        """
        This method should be able to apply the inverse of the gains to an array at full time-
        frequency resolution. Should return the input array at full resolution after the application
        of the inverse gains.
        """

        if corr_vis is None:
            corr_vis = np.empty_like(resid_vis)

        for ind in xrange(self.n_terms):  
            term = self.jones_terms[ind]

            if term.dd_term:
                break

            term.apply_inv_gains(resid_vis, corr_vis)

            resid_vis[:] = corr_vis[:]

        return corr_vis

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

    def update_term(self):
        """
        This function will update the iteration count on the relevant element of the Jones chain.
        It will also handle updating the active Jones term. Ultimately, this should handle any
        complicated convergence/term switching functionality.
        """

        def next_term():
            if self.active_term.has_converged:
                self.active_index = (self.active_index + 1)%self.n_terms
                next_term()
                return False
            else:
                return True

        check_iters = next_term()

        if (self.iters%self.term_iters) == 0 and self.iters != 0 and check_iters:
            self.active_index = (self.active_index + 1)%self.n_terms
            next_term()

        self.iters += 1

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
        return np.all([term.has_converged for term in self.jones_terms])

    @property
    def has_stalled(self):
        return np.all([term.has_stalled for term in self.jones_terms])

    @has_stalled.setter   
    def has_stalled(self, value):
        self.active_term.has_stalled = value

    @property
    def update_type(self):
        return self.active_term.update_type

    @property
    def dd_term(self):
        return self.active_term.dd_term

    @property
    def term_iters(self):
        return self.active_term.term_iters

    class Factory(MasterMachine.Factory):
        """
        Note that a ChainMachine Factory expects a list of jones options (one dict per Jones term), not a single dict.
        """
        def __init__(self, machine_cls, grid, double_precision, apply_only, global_options, jones_options):
            MasterMachine.Factory.__init__(self, machine_cls, grid, double_precision, apply_only, global_options,
                                           jones_options, jones_label="chain")


        def init_solutions(self):
            for opts in self.jones_options:
                label = opts["label"]
                self._init_solutions(label, self._make_filename(opts["load-from"]),
                                            not self.apply_only and opts["solvable"] and self._make_filename(opts["save-to"]),
                                     Complex2x2Gains.exportable_solutions(label))



                # def alternate_jh(self, model_arr, gains, blah, t_int, f_int):

    #     n_dir, n_mod, n_t, n_f, n_ant, n_ant, n_cor, n_cor = model_arr.shape

    #     jh = np.zeros_like(model_arr)

    #     for d in xrange(n_dir):
    #         rd = d%gains.shape[0]
    #         print rd
    #         for i in xrange(n_mod):
    #             for t in xrange(n_t):
    #                 rt = t//t_int
    #                 for f in xrange(n_f):
    #                     rf = f//f_int
    #                     for aa in xrange(n_ant):
    #                         for ab in xrange(n_ant):
    #                             jh[d,i,t,f,aa,ab,:] = gains[rd,rt,rf,aa,:].dot(model_arr[d,i,t,f,aa,ab,:])

    #     return jh

    # def alternate_residual(self, obser_arr, model_arr, resid_arr):

    #     n_dir, n_mod, n_t, n_f, n_ant, n_ant, n_cor, n_cor = model_arr.shape

    #     m = model_arr.copy()

    #     for ind in xrange(self.n_terms-1, -1, -1):
    #         term = self.jones_terms[ind]
    #         for d in xrange(n_dir):
    #             rd = d%term.shape[0]
    #             for i in xrange(n_mod):
    #                 for t in xrange(n_t):
    #                     rt = t//t_int
    #                     for f in xrange(n_f):
    #                         rf = f//f_int
    #                         for aa in xrange(n_ant):
    #                             for ab in xrange(n_ant):
    #                                 m[d,i,t,f,aa,ab,:] = term.gains[rd,rt,rf,aa,:].dot(m[d,i,t,f,aa,ab,:]).dot(term.gains[rd,rt,rf,ab,:].T.conj())

    #     return resid_arr - np.sum(m, axis=0, keepdims=True)