# CubiCal: a radio interferometric calibration suite
# (c) 2017 Rhodes University & Jonathan S. Kenyon
# http://github.com/ratt-ru/CubiCal
# This code is distributed under the terms of GPLv2, see LICENSE.md for details
from cubical.machines.abstract_machine import MasterMachine
from cubical.machines.complex_2x2_machine import Complex2x2Gains
from cubical.machines.complex_W_2x2_machine import ComplexW2x2Gains
import numpy as np
import cubical.kernels

from cubical.tools import logger
import machine_types
from cubical.flagging import FL
log = logger.getLogger("jones_chain")

class JonesChain(MasterMachine):
    """
    This class implements a gain machine for an arbitrary chain of Jones matrices. Most of its
    functionality is consistent with a complex 2x2 solver - many of its methods mimic those of the 
    underlying complex 2x2 machines.
    """

    def __init__(self, label, data_arr, ndir, nmod, times, frequencies, chunk_label, jones_options):
        """
        Initialises a chain of complex 2x2 gain machines.
        
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
            frequencies (np.ndarray):
                Frequencies for the data being processsed.
            jones_options (dict): 
                Dictionary of options pertaining to the chain. 
        """
        from cubical.main import UserInputError
        # This instantiates the number of complex 2x2 elements in our chain. Each element is a
        # gain machine in its own right - the purpose of this machine is to manage these machines
        # and do the relevant fiddling between parameter updates. When combining DD terms with
        # DI terms, we need to be initialise the DI terms using only one direction - we do this with
        # slicing rather than summation as it is slightly faster.
        self.jones_terms = []
        self.num_left_di_terms = 0  # how many DI terms are there at the left of the chain
        seen_dd_term = False
        for iterm, term_opts in enumerate(jones_options['chain']):
            jones_class = machine_types.get_machine_class(term_opts['type'])
            if jones_class is None:
                raise UserInputError("unknown Jones class '{}'".format(term_opts['type']))
            if jones_class not in (Complex2x2Gains, ComplexW2x2Gains) and term_opts['solvable']:
                raise UserInputError("only complex-2x2 or robust-2x2 terms can be made solvable in a Jones chain")
            term = jones_class(term_opts["label"], data_arr, ndir, nmod, times, frequencies, chunk_label, term_opts)
            self.jones_terms.append(term)
            if term.dd_term:
                seen_dd_term = True
            elif not seen_dd_term:
                self.num_left_di_terms = iterm


        MasterMachine.__init__(self, label, data_arr, ndir, nmod, times, frequencies,
                               chunk_label, jones_options)

        self.cychain  = cubical.kernels.import_kernel("cychain")
        # kernel used for compute_residuals and such
        self.cykernel = Complex2x2Gains.get_full_kernel(jones_options, diag_gains=self.is_diagonal)

        self.n_dir, self.n_mod = ndir, nmod
        _, self.n_tim, self.n_fre, self.n_ant, self.n_ant, self.n_cor, self.n_cor = data_arr.shape


        self.n_terms = len(self.jones_terms)
        # make list of number of iterations per solvable term
        # If not specified, just use the maxiter setting of each term
        # note that this list is updated as we converge, so make a copy
        term_iters = jones_options['sol']['term-iters']
        if not term_iters:
            self.term_iters = [term.maxiter for term in self.jones_terms if term.solvable]
        elif type(term_iters) is int:
            self.term_iters = [term_iters]
        elif isinstance(term_iters, (list, tuple)):
            self.term_iters = list(term_iters)
        else:
            raise UserInputError("invalid term-iters={} setting".format(term_iters))

        self.solvable = bool(self.term_iters)

        # setup first solvable term in chain
        self.active_index = -1
        self._next_chain_term()

        # this list accumulates the per-term convergence status strings
        self._convergence_states = []
        # True when the last active term has had its convergence status queried
        self._convergence_states_finalized = False

        self.cached_model_arr = self._r = self._m = None

    @classmethod
    def determine_diagonality(cls, options):
        """Returns true if the machine class, given the options, represents a diagonal gain"""
        from cubical.main import UserInputError
        diagonal = True
        for term_opts in options['chain']:
            jones_class = machine_types.get_machine_class(term_opts['type'])
            if jones_class is None:
                raise UserInputError("unknown Jones class '{}'".format(term_opts['type']))
            diagonal = diagonal and jones_class.determine_diagonality(term_opts)
        return diagonal

    @classmethod
    def determine_allocators(cls, options):
        kernel = Complex2x2Gains.get_full_kernel(options, diag_gains=cls.determine_diagonality(options))
        return kernel.allocate_vis_array, kernel.allocate_flag_array, kernel.allocate_gain_array

    def precompute_attributes(self, data_arr, model_arr, flags_arr, inv_var_chan):
        """Precomputes various stats before starting a solution"""
        MasterMachine.precompute_attributes(self, data_arr, model_arr, flags_arr, inv_var_chan)
        for term in self.jones_terms:
            if term.solvable:
                term.precompute_attributes(data_arr, model_arr, flags_arr, inv_var_chan)

    def export_solutions(self):
        """ Saves the solutions to a dict of {label: solutions,grids} items. """

        soldict = {}
        # prefix jones label to solution name
        for term in self.jones_terms:
            if term.solvable:
                for label, sol in term.export_solutions().iteritems():
                    soldict["{}:{}".format(term.jones_label, label)] = sol
        soldict['prefixed'] = True

        return soldict

    def importable_solutions(self):
        """ Returns a dictionary of importable solutions for the chain. """

        soldict = {}
        for term in self.jones_terms:
            soldict.update(term.importable_solutions())

        return soldict

    def import_solutions(self, soldict):
        """
        Loads solutions from a dict. This should not be called -- _load_solutions()
        below should rather call import_solutions() on all the chain terms.
        """
        raise RuntimeError("This method cannot be called on a Jones chain. This is a bug.")

    def _load_solutions(self, init_sols):
        """
        Helper method invoked by Factory.create_machine() to import existing solutions into machine.
        
        In the case of a chain, we invoke this method on every member.
        """
        for term in self.jones_terms:
            term._load_solutions(init_sols)

    #@profile
    def compute_js(self, obser_arr, model_arr):
        """
        This function computes the (J\ :sup:`H`\J)\ :sup:`-1` and J\ :sup:`H`\R terms of the GN/LM 
        method. This method is more complicated than a more conventional gain machine. The use of
        a chain means there are additional terms which need to be considered when computing the 
        parameter updates.

        Args:
            obser_arr (np.ndarray): 
                Shape (n_mod, n_tim, n_fre, n_ant, n_ant, n_cor, n_cor) array containing the 
                observed visibilities.
            model_arr (np.ndrray): 
                Shape (n_dir, n_mod, n_tim, n_fre, n_ant, n_ant, n_cor, n_cor) array containing the 
                model visibilities.

        Returns:
            3-element tuple
                
                - J\ :sup:`H`\R (np.ndarray)
                - (J\ :sup:`H`\J)\ :sup:`-1` (np.ndarray)
                - Count of flags raised (int)     
        """     

        n_dir, n_tint, n_fint, n_ant, n_cor, n_cor = self.active_term.gains.shape

        if self.last_active_index!=self.active_index or self.iters==1:
            self.cached_model_arr = cached_model_arr = np.empty_like(model_arr)
            np.copyto(cached_model_arr, model_arr)

            for ind in xrange(self.n_terms - 1, self.active_index, -1):
                term = self.jones_terms[ind]
                term.apply_gains(cached_model_arr)

            # collapse direction axis, if current term is non-DD
            if not self.active_term.dd_term and self.n_dir>1:
                self.cached_model_arr = np.empty_like(model_arr[0:1,...])
                np.sum(cached_model_arr, axis=0, keepdims=True, out=self.cached_model_arr)

            self.jh = np.empty_like(self.cached_model_arr)

            jhr_shape = [n_dir if self.active_term.dd_term else 1, self.n_tim, self.n_fre, n_ant, n_cor, n_cor]

            self._jhr = self.active_term.allocate_gain_array(jhr_shape, dtype=obser_arr.dtype)

            jhrint_shape = [n_dir, n_tint, n_fint, n_ant, n_cor, n_cor]

            self._jhrint = self.active_term.allocate_gain_array(jhrint_shape, dtype=self._jhr.dtype)
            self._jhj = np.empty_like(self._jhrint)
            self._jhjinv =  np.empty_like(self._jhrint)

        np.copyto(self.jh, self.cached_model_arr)

        for ind in xrange(self.active_index, -1, -1):
            term = self.jones_terms[ind]
            self.cychain.cycompute_jh(self.jh, term.gains, *term.gain_intervals)
            
        if n_dir > 1:
            if self._r is None:
                self._r = np.empty_like(obser_arr)
            r = self.compute_residual(obser_arr, model_arr, self._r)
        else:
            r = obser_arr

        self._jhr.fill(0)

        self.active_term.cykernel.cycompute_jhr(self.jh, r, self._jhr, 1, 1)

        for ind in xrange(0, self.active_index, 1):
            term = self.jones_terms[ind]
            g_inv, gh_inv, flag_counts = term.get_inverse_gains()
            self.cychain.cyapply_left_inv_jones(self._jhr, g_inv, *term.gain_intervals)

        self._jhrint.fill(0)
        self.cychain.cysum_jhr_intervals(self._jhr, self._jhrint, *self.active_term.gain_intervals)

        self._jhj.fill(0)
        self.active_term.cykernel.cycompute_jhj(self.jh, self._jhj, *self.active_term.gain_intervals)

        flag_count = self.active_term.cykernel.cycompute_jhjinv(self._jhj, self._jhjinv,
                                                    self.active_term.gflags, self.active_term.eps, FL.ILLCOND)
        return self._jhrint, self._jhjinv, flag_count

    def implement_update(self, jhr, jhjinv):
        return self.active_term.implement_update(jhr, jhjinv)

    def accumulate_gains(self, dd=True):
        """
        This function returns the product of all the gains in the chain, at full TF resolution,
        for all directions (dd=True), or just the 0th direction.

        Args:
            dd (bool):
                Accumulate per-direction gains, if available. If False, only the
                first direction is used.

        Returns:
            A tuple of gains,conjugate gains
        """
        ndir = self.n_dir if dd else 1
        gains = self.jones_terms[0].allocate_gain_array([ndir, self.n_tim, self.n_fre, self.n_ant, self.n_cor, self.n_cor],
                                                        self.dtype)
        g0 = self.jones_terms[0]._gainres_to_fullres(self.jones_terms[0].gains, tdim_ind=1)
        if ndir > 1 and g0.shape[0] == 1:
            g0 = g0.reshape(g0.shape[1:])[np.newaxis,...]
        elif ndir == 1 and g0.shape[0] > 1:
            g0 = g0[:1,...]
        gains[:] = g0
        for term in self.jones_terms[1:]:
            term.cykernel.cyright_multiply_gains(gains, term.gains, *term.gain_intervals)

        # compute conjugate gains
        gh = np.empty_like(gains)
        np.conj(gains.transpose(0, 1, 2, 3, 5, 4), gh)

        return gains, gh

    def accumulate_inv_gains(self):
        """
        This function returns the inverse of the product of all the non-DD gains in the chain, at full TF resolution,
        for direction 0

        Returns:
            A tuple of gains,conjugate gains,flag_count (if flags raised in inversion)
        """
        gains = self.jones_terms[-1].allocate_gain_array([1, self.n_tim, self.n_fre, self.n_ant, self.n_cor, self.n_cor],
                                                  self.dtype)
        init = False
        fc0 = 0
        # flip order of jones terms for inverse
        for term in self.jones_terms[::-1]:
            if not term.dd_term:
                g, _, fc = term.get_inverse_gains()
                g = term._gainres_to_fullres(g, tdim_ind=1)
                fc0 += fc
                if init:
                    term.cykernel.cyright_multiply_gains(gains, g[:1,...], *term.gain_intervals)
                else:
                    init = True
                    gains[:] = term._gainres_to_fullres(g[:1,...], tdim_ind=1)

        # compute conjugate gains
        gh = np.empty_like(gains)
        np.conj(gains.transpose(0, 1, 2, 3, 5, 4), gh)

        return gains, gh, fc


    #@profile
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
        g, gh = self.accumulate_gains()
        np.copyto(resid_arr, obser_arr)
        self.cykernel.cycompute_residual(model_arr, g, gh, resid_arr, 1, 1)

        return resid_arr

    def apply_inv_gains(self, resid_vis, corr_vis=None):
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

        if corr_vis is None:
            corr_vis = np.empty_like(resid_vis)

        g, gh, flag_count = self.accumulate_inv_gains()

        self.cykernel.cycompute_corrected(resid_vis, g, gh, corr_vis, 1, 1)

        return corr_vis, flag_count

    def apply_gains(self, vis):
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
        g, gh = self.accumulate_gains()
        self.cykernel.cyapply_gains(vis, g, gh, 1, 1)
        return vis

    def check_convergence(self, min_delta_g):
        """
        Updates the convergence info of the current time-frequency chunk. 

        Args:
            min_delta_g (float):
                Threshold for the minimum change in the gains - convergence criterion.
        """
        return self.active_term.check_convergence(min_delta_g)

    def restrict_solution(self):
        """
        Restricts the solutions by, for example, selecting a reference antenna or taking only the 
        amplitude. 
        """
        return self.active_term.restrict_solution()

    def flag_solutions(self, flags_arr, final=False):
        """ Flags gain solutions."""
        # Per-iteration flagging done on the active term, final flagging is done on all terms.
        if final:
            return any([ term.flag_solutions(flags_arr, True) for term in self.jones_terms if term.solvable ])
        else:
            return self.active_term.flag_solutions(flags_arr, False)

    def num_gain_flags(self, mask=None):
        return self.active_term.num_gain_flags(mask)

    def _next_chain_term(self):
        while True:
            if not self.term_iters:
                return False
            previous_term = self.active_term
            # clear converged and stalled properties
            self.active_index = (self.active_index + 1) % self.n_terms
            if self.active_term.solvable:
                self.active_term.maxiter = self.term_iters.pop(0)
                if not self.active_term.maxiter:
                    print>> log(1), "skipping term {}: 0 term iters specified".format(self.active_term.jones_label)
                    continue
                self.active_term.iters = 0
                self._convergence_states_finalized = False
                if previous_term:
                    previous_term.has_converged = previous_term.has_stalled = False
                self.active_term.has_converged = self.active_term.has_stalled = False
                print>> log(1), "activating term {}".format(self.active_term.jones_label)
                return True
            else:
                print>> log(1), "skipping term {}: non-solvable".format(self.active_term.jones_label)


    def next_iteration(self):
        """
        Updates the iteration count on the relevant element of the Jones chain. It will also handle 
        updating the active Jones term. Ultimately, this should handle any complicated 
        convergence/term switching functionality.
        """

        self.last_active_index = self.active_index
        major_step = False

        if self.active_term.has_converged or self.active_term.has_stalled:
            print>>log(1),"term {} {} ({} iters): {}".format(self.active_term.jones_label,
                        "converged" if self.active_term.has_converged else "stalled",
                        self.active_term.iters, self.active_term.final_convergence_status_string)
            self._convergence_states.append(self.active_term.final_convergence_status_string)
            self._convergence_states_finalized = True
            self._next_chain_term()
            major_step = True

        self.active_term.next_iteration()

        return MasterMachine.next_iteration(self)[0], major_step

    def compute_chisq(self, resid_arr, inv_var_chan):
        """Computes chi-square using the active chain term"""
        return self.active_term.compute_chisq(resid_arr, inv_var_chan)

    @property
    def has_valid_solutions(self):
        """Gives corresponding property of the active chain term"""
        return all([term.has_valid_solutions for term in self.jones_terms if term.solvable])

    @property
    def jones_label(self):
        return self.active_term.jones_label

    @property
    def num_converged_solutions(self):
        """Gives corresponding property of the active chain term"""
        return self.active_term.num_converged_solutions

    @property
    def active_term(self):
        return self.jones_terms[self.active_index]

    @property
    def dd_term(self):
        """Gives corresponding property of the active chain term"""
        return any([ term.dd_term for term in self.jones_terms ])

    @property
    def iters(self):
        """Gives corresponding property of the active chain term"""
        return self.active_term.iters

    @iters.setter
    def iters(self, value):
        """Sets corresponding property of the active chain term"""
        self.active_term.iters = value

    @property
    def maxiter(self):
        """Gives corresponding property of the active chain term"""
        return self.active_term.maxiter

    @maxiter.setter
    def maxiter(self, value):
        """Sets corresponding property of the active chain term"""
        self.active_term.maxiter = value

    @property
    def conditioning_status_string(self):
        return "; ".join([term.conditioning_status_string for term in self.jones_terms])

    @property
    def current_convergence_status_string(self):
        """Current status is reported from the active term"""
        return self.active_term.current_convergence_status_string

    @property
    def final_convergence_status_string(self):
        """Final status is reported from all terms"""
        if not self._convergence_states_finalized:
            self._convergence_states.append(self.active_term.final_convergence_status_string)
            self._convergence_states_finalized = True
        return "; ".join(self._convergence_states)

    @property
    def has_converged(self):
        # Chain has converged when term_iters is empty -- since we take off an element each time we converge a term
        return not self.solvable or \
               ( self.active_term.has_converged and not self.term_iters )
        #return np.all([term.has_converged for term in self.jones_terms])

    @property
    def has_stalled(self):
        return self.active_term.has_stalled and not self.term_iters

    @has_stalled.setter
    def has_stalled(self, value):
        self.active_term.has_stalled = value

    @property
    def epsilon(self):
        return self.active_term.epsilon

    @property
    def delta_chi(self):
        return self.active_term.delta_chi


    class Factory(MasterMachine.Factory):
        """
        Note that a ChainMachine Factory expects a list of jones options (one dict per Jones term), not a single dict.
        """
        def __init__(self, machine_cls, grid, double_precision, apply_only, global_options, jones_options):
            # manufacture dict of "Jones options" for the outer chain
            opts = dict(label="chain", solvable=not apply_only, sol=global_options['sol'], chain=jones_options)
            self.chain_options = jones_options
            MasterMachine.Factory.__init__(self, machine_cls, grid, double_precision, apply_only,
                                           global_options, opts)

        def init_solutions(self):
            for opts in self.chain_options:
                label = opts["label"]
                self._init_solutions(label,
                                     self.make_filename(opts["xfer-from"], label) or
                                     self.make_filename(opts["load-from"], label),
                                     bool(opts["xfer-from"]),
                                     self.solvable and opts["solvable"] and self.make_filename(opts["save-to"], label),
                                     Complex2x2Gains.exportable_solutions())

        def determine_allocators(self):
            return self.machine_class.determine_allocators(self.jones_options)

