# CubiCal: a radio interferometric calibration suite
# (c) 2017 Rhodes University & Jonathan S. Kenyon
# http://github.com/ratt-ru/CubiCal
# This code is distributed under the terms of GPLv2, see LICENSE.md for details
from __future__ import print_function
from six import add_metaclass
from abc import ABCMeta, abstractmethod, abstractproperty
import numpy as np
from numpy.ma import masked_array
import traceback

from cubical import param_db
from cubical.database.casa_db_adaptor import casa_db_adaptor

from cubical.tools import logger, ModColor

log = logger.getLogger("gain_machine")


@add_metaclass(ABCMeta)
class MasterMachine:
    """
    This is a base class for all solution machines. It is completely generic and lays out the basic
    requirements for all machines.
    
    It also provides a Factory class that takes care of creating machines and interfacing with 
    solution tables on disk.
    """
    __user_warnings = []

    def __init__(self, jones_label, data_arr, ndir, nmod, times, freqs, chunk_label, options, diagonal=None):
        """
        Initializes a gain machine.
        
        All supplied arguments are copied to attributes of the same name. In addition, the following
        attributes or properties are populated. These may be considered part of the official machine interface.
            
            - solvable:
                from options['solvable'], default False
            - dd_term:
                from options['dd-term'], default False
            - n_dir, n_mod, n_tim, n_fre, n_ant, n_cor:
                problem dimensions
            - dtype:
                complex dtype of data/model (complex64 or complex128)
            - ftype:
                corresponding float dtype (float32 or float64)
            - iters:
                iteration counter, reset to 0.
            - maxiters:
                from options['max-iters'], default 0
        
        Args:
            jones_label (str):
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
            chunk_label (str):
                Label of the data chunk being processed, for messages
            options (dict):
                Dictionary of options.
            diagonal (bool or None):
                Set to False or True or False if the gains are (non)diagonal. If None, calls the
                determine_diagonality() class method instead

        """
        import cubical.kernels
        self.generics = cubical.kernels.import_kernel('generics')

        self._jones_label = jones_label
        self.chunk_label = chunk_label
        self.times = times
        self.freqs = freqs
        self.options = options
        self._is_diagonal = self.determine_diagonality(options)
        self._allocate_vis_array, self._allocate_flag_array, self._allocate_gain_array = self.determine_allocators(options)

        self.solvable = options.get('solvable')
        self._dd_term = options.get('dd-term')
        self._maxiter = options.get('max-iter', 0)

        self._diag_only = options.get('diag-only')
        self._offdiag_only = options.get('offdiag-only')

        self._prop_flags = options.get('prop-flags', 'default')

        self._epsilon = options.get('epsilon')
        self._delta_chi = options.get('delta-chi')

        self.n_dir, self.n_mod = ndir if self._dd_term else 1, nmod
        _, self.n_tim, self.n_fre, self.n_ant, self.n_ant, self.n_cor, self.n_cor = data_arr.shape

        self.dtype = data_arr.dtype
        self.ftype = data_arr.real.dtype

        self._iters = 0

    @property
    def jones_label(self):
        return self._jones_label

    @property
    def is_diagonal(self):
        """Returns true if this machine instance represents a diagonal gain term"""
        return self._is_diagonal

    @property
    def allocate_vis_array(self):
        """Returns visibility allocator function for this machine"""
        return self._allocate_vis_array

    @property
    def allocate_flag_array(self):
        """Returns flag allocator function for this machine"""
        return self._allocate_flag_array

    @property
    def allocate_gain_array(self):
        """Returns flag allocator function for this machine"""
        return self._allocate_gain_array

    @classmethod
    def determine_allocators(cls, options):
        """
        Class method. Given a machine class and a set of options, returns allocator
        functions appropriate to this machine type.

        Returns:
            tuple:
                - visibility array allocator
                - flag array allocator
                - gain array allocator
        """
        return NotImplementedError

    @classmethod
    def determine_diagonality(cls, options):
        """
        Class method. Given a machine class and a set of options, returns True if the machine is
        going to represent a diagonal-only gain term.
        """
        return NotImplementedError

    @property
    def dd_term(self):
        """This property is true if the machine represents a direction-dependent gain"""
        return self._dd_term

    @property
    def diag_only(self):
        """This property is true if the machine solves using diagonal visibilities only"""
        return self._diag_only

    @property
    def offdiag_only(self):
        """This property is true if the machine solves using off-diagonal visibilities only"""
        return self._offdiag_only

    @property
    def epsilon(self):
        return self._epsilon

    @property
    def delta_chi(self):
        return self._delta_chi

    @property
    def propagates_flags(self):
        return self._prop_flags == "always" or self._prop_flags == "any" or \
               (self._prop_flags == "default" and not self.dd_term)

    @property
    def propagates_anydir_flags(self):
        return self._prop_flags == "any"

    @property
    def maxiter (self):
        """This property gives the max number of iterations"""
        return self._maxiter

    @maxiter.setter
    def maxiter (self, value):
        """Sets max number of iterations"""
        self._maxiter = value

    @property
    def iters (self):
        """This property gives the current iteration counter."""
        return self._iters

    @iters.setter
    def iters(self, value):
        """Sets current iteration counter"""
        self._iters = value

    @abstractmethod
    def compute_js(self, obser_arr, model_arr):
        """
        This method is expected to compute (J\ :sup:`H`\J)\ :sup:`-1` and J\ :sup:`H`\R. 
        
        Args:
            model_arr (np.ndrray): 
                Shape (n_dir, n_mod, n_tim, n_fre, n_ant, n_ant, n_cor, n_cor) array containing the 
                model visibilities.
            obser_arr (np.ndarray): 
                Shape (n_mod, n_tim, n_fre, n_ant, n_ant, n_cor, n_cor) array containing the 
                observed visibilities.            

        
        Returns:
            3-element tuple
                
                - J\ :sup:`H`\R (np.ndarray)
                - (J\ :sup:`H`\J)\ :sup:`-1` (np.ndarray)
                - Count of flags raised (int)     
        """

        return NotImplementedError

    @abstractmethod
    def implement_update(self, jhr, jhjinv):
        """
        Internal method implementing a parameter update. The standard compute_update() implementation 
        calls compute_js() and _implement_update() at each iteration step.
        
        Args:
            jhr (np.ndarray):
                J\ :sup:`H`\R term
            jhjinv (np.ndarray):
                (J\ :sup:`H`\J)\ :sup:`-1` term
        """
        return NotImplementedError

    def compute_update(self, model_arr, obser_arr):
        """
        This method is expected to compute the parameter update. 
        
        The standard implementation simply calls compute_js() and implement_update(), but subclasses are free to 
        override.

        Args:
            model_arr (np.ndarray): 
                Shape (n_dir, n_mod, n_tim, n_fre, n_ant, n_ant, n_cor, n_cor) array containing 
                model visibilities. 
            obser_arr (np.ndarray): 
                Shape (n_mod, n_tim, n_fre, n_ant, n_ant, n_cor, n_cor) array containing observed 
                visibilities. 
        """
        jhr, jhjinv, flag_count = self.compute_js(obser_arr, model_arr)

        self.implement_update(jhr, jhjinv)

        return flag_count

    @abstractmethod
    def compute_residual(self, obser_arr, model_arr, resid_arr, require_full=True):
        """
        This method should compute the residual at the the full time-frequency resolution of the
        data. Must populate resid_arr with the values of the residual. Function signature must be 
        consistent with the one defined here.

        Args:
            obser_arr (np.ndarray): 
                Shape (n_mod, n_tim, n_fre, n_ant, n_ant, n_cor, n_cor) array containing observed 
                visibilities. 
            model_arr (np.ndarray): 
                Shape (n_dir, n_mod, n_tim, n_fre, n_ant, n_ant, n_cor, n_cor) array containing 
                model visibilities.
            resid_arr (np.ndarray):
                Shape (n_tim, n_fre, n_ant, n_ant, n_cor, n_cor) array in which to place the 
                residual values.
            require_full (bool):
                If True, a full 2x2 residual is required. If False, only the terms used in the solution
                (e.g. if a solution is diagonal, only the diagonals) are required.
        """

        return NotImplementedError

    @abstractmethod
    def apply_inv_gains(self, obser_arr, corr_vis=None, full2x2=True, direction=None):
        """
        This method should be able to apply the inverse of the gains associated with the gain
        machines to an array at full time-frequency resolution. Should populate an input array with
        the result or return a new array. Function signature must be consistent with the one defined
        here.

        Args:
            obser_arr (np.ndarray):
                Shape (n_tim, n_fre, n_ant, n_ant, n_cor, n_cor) array containing observed 
                visibilities.
            corr_vis (np.ndarray or None, optional):
                Shape (n_tim, n_fre, n_ant, n_ant, n_cor, n_cor) array to fill with the corrected 
                visibilities.
            full2x2 (bool):
                If True, gains should be applied to the full 2x2 matrix. If False, only the terms used in the solution
                (e.g. the diagonals) are required.
            direction (int or None):
                If None, only DI terms (in a chain) are applied. If set to a direction number,
                also applies DD terms for that direction.

        Returns:
            2-element tuple
                
                - Corrected visibilities (np.ndarray)
                - Flags raised (int)
        """
        
        return NotImplementedError

    @abstractmethod
    def apply_gains(self, model_arr, full2x2=True, dd_only=False):
        """
        This method should be able to apply the gains associated with the gain
        to an array at full time-frequency resolution. 
        Function signature must be consistent with the one defined here.

        Args:
            model_arr (np.ndarray):
                Shape (n_dir, n_mod, n_tim, n_fre, n_ant, n_ant, n_cor, n_cor) array containing model visibilities.
            full2x2 (bool):
                If True, gains should be applied to the full 2x2 matrix. If False, only the terms used in the solution
                (e.g. the diagonals) are required.
            dd_only (bool):
                If True, only DD terms are applied, beginning with the leftmost. Not implemented for now.
        """

        return NotImplementedError

    @property
    def has_valid_solutions(self):
        """
        This property returns True if the gain machine has any valid solutions defined. (The opposite would be
        true if e.g. they were all flagged).
        """
        return False

    @property
    def num_solutions(self):
        """
        This property gives the number of solutions (e.g. intervals) defined by the machine
        """
        return 0

    @property
    def num_converged_solutions(self):
        """
        This property gives the number of currently converged solutions defined by the machine
        """
        return 0

    def update_equation_counts(self, unflagged):
        """
        Sets up equation counters and normalization factors. Sets up the following attributes:
            - eqs_per_record
                integer: gives the nominal number of equations per a visibility record, so e.g.
                would be 8 for a full 2x2 complex correlation matrix
            - eqs_per_tf_slot (np.ndarray):
                Shape (n_tim, n_fre) array containing a count of equations per time-frequency slot.
            - eqs_per_antenna (np.ndarray)
                Shape (n_ant, ) array containing a count of equations per antenna.

        Also sets up corresponding chi-sq normalization factors.

        Normally called form precompute_attributes() to set things up. Should also be called every time
        the flags change (otherwise chi-sq normalization will go off)

        Args:
            unflagged (np.ndarray):
                Shape (n_tim, n_fre, n_ant, n_ant) bool array indicating valid (not flagged) slots
        """

        # equations per visibility slot. We have two (real/imag, or normal and conjugate) per model
        # and per correlation product. This is just a normalization factor used in the bookkeeping.
        self.eqs_per_record = 2 * self.n_mod * self.n_cor
        # full 2x2 as opposed to diag-only or off-diag-only? Multiply by 2
        if not self.diag_only or self.offdiag_only:
            self.eqs_per_record *= self.n_cor

        # (n_ant) vector containing the number of valid equations per antenna (for any direction)

        self.eqs_per_antenna = np.sum(unflagged, axis=(0, 1, 2)) * self.eqs_per_record

        # (n_tim, n_fre) array containing number of valid equations for each time/freq slot

        self.eqs_per_tf_slot = np.sum(unflagged, axis=(-1, -2)) * self.eqs_per_record


        with np.errstate(invalid='ignore', divide='ignore'):
            self._chisq_tf_norm_factor = 1./self.eqs_per_tf_slot
        self._chisq_tf_norm_factor[self.eqs_per_tf_slot==0] = 0

        toteq = np.sum(self.eqs_per_tf_slot)
        self._chisq_norm_factor = 1./toteq if toteq else 0

        self._chisq_tf_norm_factor_0 = self._chisq_tf_norm_factor
        self._chisq_norm_factor_0 = self._chisq_norm_factor


    def compute_chisq(self, resid_arr, inv_var_chan, require_full=True):
        """
        Computes chi-squared statistics based on given residuals.
        Args:
            resid_arr (np.ndarray):
                Shape (n_mod, n_tim, n_fre, n_ant, n_ant, n_cor, n_cor) array containing residuals.
            inv_var_chan (np.ndarray)
                Shape (nfreq,) array of 1/sigma^2 per channel

        Returns:
            3-tuple consisting of
            
             - chisq (np.ndarray):
                (n_tim, n_fre, n_ant) array of chi-squared per antenna and time/frequency slot
             - chisq_per_tf_slot  (np.ndarray):
                (n_tim, n_fre) array of chi-squared per time/frequency slot
             - chisq_tot (float):
                overall chi-squared value for the entire chunk
        """

        # Chi-squared is computed by summation over antennas, correlations and intervals. Sum over
        # time intervals, antennas and correlations first. Normalize by per-channel variance and
        # finally sum over frequency intervals.

        chisq = np.zeros(resid_arr.shape[1:4], np.float64)
        if self.diag_only:
            self.generics.compute_chisq_diag(resid_arr, chisq)
        elif self.offdiag_only:
            self.generics.compute_chisq_offdiag(resid_arr, chisq)
        else:
            self.generics.compute_chisq(resid_arr, chisq)

        # Normalize this by the per-channel variance.

        chisq *= inv_var_chan[np.newaxis, :, np.newaxis]

        # Collapse chisq to chi-squared per time-frequency slot and overall chi-squared

        chisq_per_tf_slot = np.sum(chisq, axis=-1) * self._chisq_tf_norm_factor

        chisq_tot = np.sum(chisq) * self._chisq_norm_factor

        return chisq, chisq_per_tf_slot, chisq_tot

    def precompute_attributes(self, data_arr, model_arr, flags_arr, inv_var_chan):
        """
        This method is called before starting a solution. The base version computes a variety of useful 
        parameters regarding the conditioning and degrees of freedom of the current time-frequency chunk. 
        Subclasses can redefine this to precompute other useful stuff (i.e. things that do not vary with 
        iteration).

        Args:
            data_arr (np.ndarray):
                Shape (n_mod, n_tim, n_fre, n_ant, n_ant, n_cor, n_cor) array containing observed visibilities.
            model_arr (np.ndarray):
                Shape (n_dir, n_mod, n_tim, n_fre, n_ant, n_ant, n_cor, n_cor) array containing model visibilities.
            flags_arr (np.ndarray):
                Shape (n_tim, n_fre, n_ant, n_ant) array containing integer flags
            inv_var_chan (np.ndarray)
                Shape (nfreq,) array of 1/sigma^2 per channel
                
        Returns:
            bool np.ndarray, shape (n_tim, n_fre, n_ant, n_ant), indicating the inverse of flags
        """
        unflagged = flags_arr==0
        self.update_equation_counts(unflagged)
        return unflagged

    @abstractmethod
    def check_convergence(self, min_delta_g):
        """
        This method should check the gain solutions for convergence.

        Args:
            min_delta_g (float):
                Threshold for the minimum change in the gains - convergence criterion.
        """

        return NotImplementedError

    @abstractmethod
    def flag_solutions(self, flag_arr, final=0):
        """
        This method allows the machine to flag gains solutions.
        When solving, it is called after each iteration (final=0), and then once again after convergence (final=1).
        When applying, it is called once after loading the solutions with final=-1
        
        This method can propagate the flags raised by the gain machine back into the data flags.
        
        Args:
            flag_arr (np.ndarray):
                Shape (n_tim, n_fre, n_ant, n_ant) array containing data flags. 
            final (int):
                -1 when loading (in apply-only mode), 0 while iterating, 1 after convergence.

        Returns:
            True if any new flags have been propagated to the data
        
        """
        return NotImplementedError

    @abstractmethod
    def num_gain_flags(self, mask=None, final=False):
        """
        This method returns the number of gains flagged, and the total number of gains.
        
        Args:
            mask (int):
                Flag mask to apply. If None, ~FL.MISSING is expected to be used
            final:
                If True, this is the final (post-solution) call, so return sum of all flags (if e.g. a chain)
 
        Returns:
            Tuple of two values
                - number of flagged gains
                - total number of gains

        """
        return NotImplementedError

    def next_iteration(self):
        """
        This method should update the current iteration counter, as well as handling any more complicated
        behaviour required for multiple Jones terms. Default version just bumps the iteration counter.
        
        Returns:
            Tuple of two values
                - value of iteration counter
                - True/False hint indicating if a "major" step was taken
        """
        self._iters += 1
        return self.iters, False

    @abstractmethod
    def restrict_solution(self, gains):
        """
        This method should perform any necessary restrictions on the solution, eg. selecting a 
        reference antenna or taking only the amplitude. Function signature must be consistent with 
        the one defined here.
        """

        return NotImplementedError

    @abstractproperty
    def has_converged(self):
        """
        This property must return the convergence status of the gain machine. 
        """
        return NotImplementedError

    @abstractproperty
    def has_stalled(self):
        """
        This property must return the convergence stall status of the gain machine. Note that it
        may be assigned to in the solver.
        """
        return NotImplementedError

    @abstractproperty
    def conditioning_status_string(self):
        """
        This property must return a conditioning status string for the gain machine, e.g.
        
        """
        return NotImplementedError

    @abstractproperty
    def current_convergence_status_string(self):
        """
        This property must return a convergence status string for the gain machine, e.g.
            "G: 20 iters, conv 60.02%, g/fl 15.00%, max update 1e-3"
        This is called while iterating.
        """
        return NotImplementedError

    @property
    def final_convergence_status_string(self):
        """
        This property must return a convergence status string for the gain machine, e.g.
            "G: 20 iters, conv 60.02%, g/fl 15.00%, max update 1e-3"
        This is called once the solution is finished, so it may report a slightly different
        status. Default version uses the current status.
        """
        return self.current_convergence_status_string


    @staticmethod
    def exportable_solutions():
        """
        Returns dict of {label: (empty_value, axes_list)} describing the types of parameters that
        this machine can export. Axes is a list of axis labels. Static method, as it is called
        before any GM is actually created. If empty_value is float or complex, global precision 
        settings will be used.
        """

        return {}

    def importable_solutions(self, grid0):
        """
        Returns dict of parameters that this machine can import, as {label: grid_dict}. Grid_dict 
        knows which grid the parameters must be interpolated onto for this machine. Called when a 
        machine has been created (so grids are available).
        """

        return {}

    @abstractmethod
    def export_solutions(self):
        """
        This method returns the solutions as a dict of {label: masked_array, grid} elements. Arrays
        are masked since solutions have associated flags. Labels must be consistent with whatever 
        exportable_solutions() returns, but not all solutions promised by exportable_solutions() 
        will actually need to be exported.
        
        Grid is a dict, defining axes on which solutions are given, e.g. {'time': vector, 
        'freq': vector}. Note that axes which are fully spanned (e.g. antenna, correlation) need 
        not be present in the grid.
        """

        return NotImplementedError

    @abstractmethod
    def import_solutions(self, solutions_dict):
        """
        This method loads the internal solutions from a dict of {label: array} elements. Labels 
        must be in importable_solutions. Array are masked since solutions have associated flags.
        Array shapes will conform to importable_solutions() results.
        """

        return NotImplementedError


    @classmethod
    def create_factory(machine_cls, *args, **kw):
        """
        This method creates a Machine Factory that will initialize the correct type of gain machine.
        This must be called via the subclasses, so that the factory gets the proper class 
        information.
        """

        return machine_cls.Factory(machine_cls, *args, **kw)

    def _load_solutions(self, init_sols, full_grid):
        """
        Helper method invoked by Factory.create_machine() to import existing solutions into machine.
        Looks for solutions corresponding to this machine's jones_label in init_sols, and
        invokes import_solutions() as appropriate.
        
        Args:
            init_sols: dict of initial solutions, given as {jones_label:(database, prefix)}
        """
        sols = {}
        # collect importable solutions from DB, interpolate
        for label, grids in self.importable_solutions(full_grid).items():
            db, prefix, interpolate = init_sols.get(self.jones_label, (None, None, False))
            name = "{}:{}".format(prefix, label)
            if db is not None:
                if name in db:
                    # check for matching grids
                    mismatches = db[name].find_mismatched_grids(**grids)
                    if mismatches:
                        # are non-interpolatable axes also missing?
                        missing_solutions = not all([interpolatable for _,_,interpolatable in mismatches])
                        # info if only interpolatable entries missing, else warning
                        if interpolate and not missing_solutions:
                            report_func = log(0).print
                        else:
                            report_func = log.warn
                        report_func("{}: {} grid not exactly matched in {}, interpolation may be required".format(
                                self.chunk_label, name, db.filename))
                        # report on axes one by one
                        for axis, values, interpolatable in mismatches:
                            report_func = log(0).print if interpolate and interpolatable else log.warn
                            report_func("    {}: {} not matched{}".format(axis,
                                                     "{} entries".format(len(values)) if len(values) > 5 else
                                                     ", ".join(map(str, values)),
                                                    (", will interpolate" if interpolate and interpolatable else "")))
                    # if interpolating, this is allowed -- otherwise crash out
                    if interpolate:
                        print("{}: interpolating {} from {}".format(self.chunk_label, name, db.filename), file=log)
                        sols[label] = sol = db[name].reinterpolate(**grids)
#                        import ipdb; ipdb.set_trace()
                    else:
                        if mismatches:
                            raise ValueError("{} does not define {} on the correct grid. Consider using "
                                             "-xfer-from rather than -load-from".format(name, db.filename))
                        print("{}: loading {} from {}".format(self.chunk_label, name, db.filename), file=log)
                        sols[label] = sol = db[name].lookup(**grids)
                    if sol.count() != sol.size:
                        print("{}: {:.2%} valid {} slots populated".format(
                            self.chunk_label, sol.count()/float(sol.size), name), file=log)
                    db[name].release_cache()
                else:
                    print("{}: {} not in {}".format(self.chunk_label, name, db.filename), file=log)
        # if anything at all was loaded from DB, import
        if sols:
            self.import_solutions(sols)

    @classmethod
    def raise_userwarning(self, level, msg, sort_index, verbosity=0, raise_once=None, color=None):
        """
        Raise user warning to be summarized after solving
        """
        MasterMachine.__user_warnings.append({
            "level": level,
            "msg": msg,
            "sort_index": sort_index,
            "raise_once": raise_once,
            "verbosity": verbosity,
            "color": color})

    @classmethod
    def collect_warnings(self):
        """
        Sorted list of user warnings raised during solving
        """
        res = sorted(MasterMachine.__user_warnings, key=lambda d: d.get("sort_index", 0))
        MasterMachine.__user_warnings = []
        return res

    class Factory(object):
        """
        A Machine Factory class is created as a singleton: it is responsible for creating and initializing 
        individual gain machines for chunks of the solution space. The purpose of a factory is to hold "global" 
        information for the solution overall (e.g. the overall time/frequency range etc.)
        
        TODO: directions must also be passed in
        """

        def __init__(self, machine_cls, grid, double_precision, apply_only, global_options, jones_options):
            """
            Initializes a factory

            Args:
                machine_cls: the class of the gain machine that will be created
                
                grid: the grid on which solutions are expected, for axes that are already known (antennas, correlations)
                  A dictionary of {axis_name:values_vector}. time/freq need not be supplied, as it is filled in chunk by chunk.
                  
                double_precision: if True, gain machines will use 64-bit float arithmetic
                
                apply_only: if True, solutions are only applied, not solved for
                
                global_options: dict of global options for CubiCal (GD)
                
                jones_options: dict of Jones term options for the specific Jones term
            """
            self.global_options = global_options
            self.jones_options = jones_options
            self.jones_label = self.jones_options['label']
            self.machine_class = machine_cls
            if apply_only:
                self.jones_options['solvable'] = False
            self.solvable = self.jones_options['solvable']
            self.grid = grid
            self.ctype = np.complex128 if double_precision else np.complex64
            self.ftype = np.float64    if double_precision else np.float32
            # dict of jones label -> param database object to init from
            self._init_sols = {}
            # dict of jones label -> param database object to save to. Multiple jones labels
            # may share the same DB (_save_sols_byname holds a unique dict)
            self._save_sols = {}
            # dict of filename -> unique param database object.
            self._save_sols_byname = {}
            # initialize solution databases
            self.init_solutions()

        def get_allocators(self):
            """
            Returns allocation functions appropriate for the class of the gain machine.
            Returns tuple of vis_allocator, flag_allocator.
            """
            return self.machine_class.get_allocators(self.jones_options)

        def init_solutions(self):
            """
            Internal method, called from constructor. Initializes solution databases.
            Default behaviour uses the _init_solutions() implementation below.
            Note that this is reimplemented in JonesChain to collect solution info from chain.
            """
            self._init_solutions(self.jones_label,
                                 self.make_filename(self.jones_options["xfer-from"]) or
                                 self.make_filename(self.jones_options["load-from"]),
                                 bool(self.jones_options["xfer-from"]),
                                 self.solvable and self.make_filename(self.jones_options["save-to"]),
                                 self.machine_class.exportable_solutions(),
                                 dd_term=bool(self.jones_options["dd-term"]))

        def make_filename(self, filename, jones_label=None):
            """
            Helper method: expands full filename a from templated filename. This uses the standard 
            str.format() function, passing in self.global_options, as well as JONES=jones_label, as keyword 
            arguments. This allows for filename templates that include strings from the global options
            dictionary, e.g. "{data[ms]}-ddid{sel[ddid]}".
            
            Args:
                filename (str): 
                    the templated filename
                jones_label (str, optional):
                    Jones matrix label, overrides self.jones_label if specified.
                
            Returns:
                str:
                    Expanded filename
                
            """
            from cubical.main import expand_templated_name
            return expand_templated_name(filename,
                                         JONES=jones_label or self.jones_label)

        def _init_solutions(self, label, load_from, interpolate, save_to, exportables, dd_term=False):
            """
            Internal helper implementation for init_solutions(): this initializes a pair of solution databases.
            
            Args:
                label (str): 
                    the Jones matrix label
                load_from (str):
                    filename of solution DB to load from. Can be empty or None.
                interpolate (bool):
                    True if solutions are allowed to be interpolated.
                save_to (str):
                    filename of solution DB to save to. Can be empty or None.
                exportables (dict):
                    Dictionary of {key: (empty_value, axis_list)} describing the solutions that will be saved.
                    As returned by exportable_solutions() of the gain machine.
                dd_term (bool):
                    True if paremeter was configured as direction-dependent
                
            """
            # init solutions from database
            if load_from:
                print("{} solutions will be initialized from {}".format(label, load_from), file=log(0, "blue"))
                if "//" in load_from:
                    filename, prefix = load_from.rsplit("//", 1)
                else:
                    filename, prefix = load_from, label
                self._init_sols[label] = param_db.load(filename), prefix, interpolate
            # create database to save to
            if save_to:
                # define parameters in DB
                for sol_label, (empty_value, axes) in exportables.items():
                    self.define_param(save_to, "{}:{}".format(label, sol_label), empty_value, axes, dd_term=dd_term)
                print("{} solutions will be saved to {}".format(label, save_to), file=log(0))

        def define_param(self, save_to, name, empty_value, axes,
                          interpolation_axes=("time", "freq"), dd_term=False):
            """
            Internal helper method for _init_solutions(). Defines a parameter to be saved.
            
            Args:
                save_to (str):
                    filename of solution DB to save to. Can be empty or None.
                name (str):
                    name of parameter to be saved
                empty_value:
                    Scalar representing an empty (default) solution, e.g. zero or unity of int, float or complex type.
                axes (iterable):
                    List of axes over which the parameter is defined, e.g. ["time", "freq", "ant1", "ant2"]
                interpolation_axes (iterable):
                    List of axes over which the parameter can be interpolated. Subset of axes.
                dd_term (bool):
                    True if paremeter was configured as direction-dependent
            """
            # init DB, if needed
            db = self._save_sols_byname.get(save_to)
            if db is None:
                selfield = self.global_options["sel"]["field"]
                assert type(selfield) is int, "Currently only supports single field data selection"
                selddid = self.global_options["sel"]["ddid"]
                #assert ((type(selddid) is list) and (all([type(t) is int for t in selddid]))) or \
                #       (type(selddid) is int) or selddid is None, "SPW should be a list of ints or int or None. This is a bug"
                meta = {"field": selfield}
                self._save_sols_byname[save_to] = db = param_db.create(save_to, metadata=meta, backup=True)
            self._save_sols[name] = db
            # work out type of empty value
            if type(empty_value) is float:
                dtype = self.ftype
            elif type(empty_value) is complex:
                dtype = self.ctype
            else:
                dtype = type(empty_value)
            # build parameter grid from predefined grid. Trailing digits are handled:
            # solution axes such as "ant", "ant1" and "ant2" will map to predefined axis "ant",
            # if this is available.
            grid = {}
            for axis in axes:
                if axis in self.grid:
                    # use saved grid, or else [0] for direction axis of non-DD parameter
                    grid[axis] = self.grid[axis] if axis != "dir" or dd_term else [0]
                elif axis[-1].isdigit() and axis[:-1] in self.grid:
                    grid[axis] = self.grid[axis[:-1]]
            return db.define_param(name, dtype, axes, empty=empty_value,
                                   interpolation_axes=interpolation_axes, grid=grid)

        def export_solutions(self, gm, subdict):
            """
            Exports a slice of solutions from a gain machine into a shared dictionary.
            We use shared memory (shared_dict.SharedDict) objects to pass solutions between
            worker (solver) processes and the I/O process which ultimately saves them. This 
            method is called on the solver process side to populate the shared dict.
            
            Args:
                gm:
                    An instance of a gain machine
                    
                subdict (shared_dict.SharedDict):
                    Shared dictionary to be populated (this is generally a subdict of some
                    larger shared dictionary, uniquely assigned to this solver process)
            """
            if not self.solvable:
                return
            sols = gm.export_solutions()
            # has the gain machine added a prefix to the names already (as the chain machine does)
            is_prefixed = sols.pop('prefixed', False)
            # populate values subdictionary
            for label, (value, grid) in sols.items():
                name = label if is_prefixed else "{}:{}".format(gm.jones_label, label)
                subdict[name] = value.data
                subdict["{}:grid__".format(name)]  = grid
                subdict["{}:flags__".format(name)] = value.mask

        def get_solution_db(self, name):
            """
            Returns (output) solution database corresponding to the named parameter
            """
            return self._save_sols[name]

        def save_solutions(self, subdict):
            """
            Saves a slice of the solutions from a dictionary to the database. This is called in the I/O process
            to actually save the solutions that are exported by export_solutions().

            Args:
                subdict (shared_dict.SharedDict):
                    Shared dictionary to be saved. This is presumed to be populated by export_solutions() above.
            """
            # add slices for all parameters
            for name in subdict.keys():
                if not name.endswith("__") and name in self._save_sols:
                    sd = subdict["{}:grid__".format(name)]
                    grids = {key: sd[key] for key in sd.keys()}
                    log(1).print("saving solutions for {}".format(name))
                    self.get_solution_db(name).add_chunk(name, masked_array(subdict[name],
                                                                       subdict[name+":flags__"]), grids)
        def close(self):
            """
            Closes all solution databases and releases various caches.
            """
            for db, prefix, _ in list(self._init_sols.values()):
                db.close()
            for db in list(self._save_sols_byname.values()):
                db.close()
            self._init_sols = {}
            self._save_sols = {}
            self._save_sols_byname = {}

        def create_machine(self, data_arr, n_dir, n_mod, times, freqs, chunk_label):
            """
            Creates a gain machine, given a model, and a time and frequency subset of the global solution space.

            Args:
                data_arr: model array, of shape (nmod,ntim,nfreq,nant,nant,ncorr,ncorr)
                times: times
                freqs: frequencies

            Returns:
                An instance of a gain machine
            """
            gm = self.machine_class(self.jones_label, data_arr, n_dir, n_mod, times, freqs,
                                    chunk_label, self.jones_options)
            gm._load_solutions(self._init_sols, self.grid)
            return gm
        
        def set_metadata(self, src):
            """
            Sets database metadata source
            
            Args:
                src: instance of cubical.data_handler
            """
            self.metadata = src.metadata
            for db in list(self._save_sols_byname.values()):
                db.export_CASA_gaintable = self.global_options["out"].get("casa-gaintables", True)
                db.set_metadata(src)

        def determine_allocators(self):
            return self.machine_class.determine_allocators(self.jones_options)
