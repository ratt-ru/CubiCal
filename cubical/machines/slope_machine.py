# CubiCal: a radio interferometric calibration suite
# (c) 2017 Rhodes University & Jonathan S. Kenyon
# http://github.com/ratt-ru/CubiCal
# This code is distributed under the terms of GPLv2, see LICENSE.md for details
from cubical.machines.parameterised_machine import ParameterisedGains
import numpy as np
from numpy.ma import masked_array
import cubical.kernels

def _normalize(x, dtype):
    """
    Helper function: normalizes array to [0,1] interval.
    """
    if len(x) > 1:
        return ((x - x[0]) / (x[-1] - x[0])).astype(dtype)
    elif len(x) == 1:
        return np.zeros(1, dtype)
    else:
        return x


class PhaseSlopeGains(ParameterisedGains):
    """
    This class implements the diagonal phase-only parameterised slope gain machine.
    """

    def __init__(self, label, data_arr, ndir, nmod, chunk_ts, chunk_fs, chunk_label, options):
        """
        Initialises a diagonal phase-slope gain machine.
        
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
            chunk_ts (np.ndarray):
                Times for the data being processed.
            chunk_fs (np.ndarray):
                Frequencies for the data being processsed.
            options (dict): 
                Dictionary of options. 
        """
        ### this kernel used for residuals etc.
        cykernel = self.get_kernel(options)

        ParameterisedGains.__init__(self, label, data_arr, ndir, nmod,
                                    chunk_ts, chunk_fs, chunk_label, options, cykernel)

        self.slope_type = options["type"]
        self.n_param = 3 if self.slope_type == "tf-plane" else 2

        self.param_shape = [self.n_dir, self.n_timint, self.n_freint, 
                            self.n_ant, self.n_param, self.n_cor, self.n_cor]
        self.slope_params = np.zeros(self.param_shape, dtype=self.ftype)
        self.posterior_slope_error = None

        self.chunk_ts = _normalize(chunk_ts, self.ftype)
        self.chunk_fs = _normalize(chunk_fs, self.ftype)

        if self.slope_type == "tf-plane":
            self.cyslope = cubical.kernels.import_kernel("cytf_plane")
            self._labels = dict(phase=0, delay=1, rate=2)
        elif self.slope_type == "f-slope":
            self.cyslope = cubical.kernels.import_kernel("cyf_slope")
            self._labels = dict(phase=0, delay=1)
        elif self.slope_type == "t-slope":
            self.cyslope = cubical.kernels.import_kernel("cyt_slope")
            self._labels = dict(phase=0, rate=1)
        else:
            raise RuntimeError("unknown machine type '{}'".format(self.slope_type))

        self._jhr0 = self._gerr = None

    @staticmethod
    def get_kernel(options):
        """Returns kernel approriate to Jones options"""
        if options['diag-diag']:
            return cubical.kernels.import_kernel('cydiag_phase_only')
        else:
            return cubical.kernels.import_kernel('cyphase_only')

    @staticmethod
    def exportable_solutions():
        """ Returns a dictionary of exportable solutions for this machine type. """

        exportables = ParameterisedGains.exportable_solutions()

        exportables.update({
            "phase": (0., ("dir", "time", "freq", "ant", "corr")),
            "delay":  (0., ("dir", "time", "freq", "ant", "corr")),
            "rate":   (0., ("dir", "time", "freq", "ant", "corr")),
            "phase.err": (0., ("dir", "time", "freq", "ant", "corr")),
            "delay.err": (0., ("dir", "time", "freq", "ant", "corr")),
            "rate.err": (0., ("dir", "time", "freq", "ant", "corr")),
        })
        
        return exportables

    def get_inverse_gains(self):
        """Returns inverse gains and inverse conjugate gains. For phase-only, conjugation is inverse"""
        gh = self.get_conj_gains()
        return gh, self.gains, 0

    def importable_solutions(self):
        """ Returns a dictionary of importable solutions for this machine type. """

        # defines solutions we can import from
        # Note that complex gain (as a derived parameter) is exported, but not imported
        return { label: self.interval_grid for label in self._labels.iterkeys() }

    def export_solutions(self):
        """ Saves the solutions to a dict of {label: solutions,grids} items. """

        solutions = ParameterisedGains.export_solutions(self)

        for label, num in self._labels.iteritems():
            solutions[label] = masked_array(self.slope_params[...,num,(0,1),(0,1)]), self.interval_grid
            if self.posterior_slope_error is not None:
                solutions[label+".err"] = masked_array(self.posterior_slope_error[..., num, :]), self.interval_grid

        return solutions

    def import_solutions(self, soldict):
        """ 
        Loads solutions from a dict. 
        
        Args:
            soldict (dict):
                Contains gains solutions which must be loaded.
        """
        
        # Note that this is inherently very flexible. For example, we can init from a solutions
        # table which only has a "phase" entry, e.g. one generated by a phase_only solver (and the
        # delay will then be left at zero).

        loaded = False
        for label, num in self._labels.iteritems():
            value = soldict.get(label)
            if value is not None:
                self.slope_params[...,num,(0,1),(0,1)] = value
                loaded = True

        if loaded:
            self.cyslope.cyconstruct_gains(self.slope_params, self.gains,
                                           self.chunk_ts, self.chunk_fs, self.t_int, self.f_int)
        

    def compute_js(self, obser_arr, model_arr):
        """
        This function computes the J\ :sup:`H`\R term of the GN/LM method. 

        Args:
            obser_arr (np.ndarray): 
                Shape (n_mod, n_tim, n_fre, n_ant, n_ant, n_cor, n_cor) array containing the 
                observed visibilities.
            model_arr (np.ndrray): 
                Shape (n_dir, n_mod, n_tim, n_fre, n_ant, n_ant, n_cor, n_cor) array containing the 
                model visibilities.

        Returns:
            np.ndarray:
                J\ :sup:`H`\R
        """

        n_dir, n_tim, n_fre, n_ant, n_cor, n_cor = self.gains.shape

        gh = self.get_conj_gains()
        jh = self.get_new_jh(model_arr)

        self.cyslope.cycompute_jh(model_arr, self.gains, jh, 1, 1)

        jhr1 = self.get_new_jhr()

        r = self.get_obs_or_res(obser_arr, model_arr)

        # use appropriate phase-only kernel (with 1,1 intervals) to compute inner JHR
        self.cykernel.cycompute_jhr(gh, jh, r, jhr1, 1, 1)

        jhr1 = jhr1.imag

        jhr_shape = [n_dir, self.n_timint, self.n_freint, n_ant, self.n_param, n_cor, n_cor]

        if self._jhr0 is None:
            self._jhr0 = self.cyslope.allocate_param_array(jhr_shape, dtype=jhr1.dtype, zeros=True)
        else:
            self._jhr0.fill(0)

        self.cyslope.cycompute_jhr(jhr1, self._jhr0, self.chunk_ts, self.chunk_fs, self.t_int, self.f_int)

        return self._jhr0, self.jhjinv, 0

    @property
    def dof_per_antenna(self):
        """This property returns the number of real degrees of freedom per antenna, per solution interval"""
        if self.slope_type=="tf-plane":
            return 6
        elif self.slope_type=="f-slope":
            return 4
        elif self.slope_type=="t-slope":
            return 4

    def implement_update(self, jhr, jhjinv):

        # variance of slope parms is diagonal of jhjinv
        diag = (0,2) if self.n_param == 2 else (0,3,5)   # weird numbering to get diagonal elements
        var_slope = jhjinv[...,(0,1),(0,1)].real[...,diag,:]
        if self.posterior_slope_error is None:
            self.posterior_slope_error = np.sqrt(var_slope)
        else:
            np.sqrt(var_slope, out=self.posterior_slope_error)

        # variance of gain is sum of slope parameter variances
        if self._gerr is None:
            self._gerr = var_slope.sum(axis=-2)
        else:
            var_slope.sum(axis=-2, out=self._gerr)

        np.sqrt(self._gerr, out=self._gerr)
        if self.posterior_gain_error is None:
            self.posterior_gain_error = np.zeros_like(self.gains, dtype=float)
        self._interval_to_gainres(self._gerr[...,0], 1, out=self.posterior_gain_error[...,0,0])
        self._interval_to_gainres(self._gerr[...,0], 1, out=self.posterior_gain_error[...,1,1])

        update = self.init_update(jhr)

        self.cyslope.cycompute_update(jhr, jhjinv, update)

        if self.iters%2 == 0:
            update *= 0.5
        
        self.slope_params += update

        self.restrict_solution()

        # Need to turn updated parameters into gains.

        self.cyslope.cyconstruct_gains(self.slope_params, self.gains, self.chunk_ts, self.chunk_fs, self.t_int, self.f_int)

    def restrict_solution(self):
        """
        Restricts the solution by invoking the inherited restrict_soultion method and applying
        any machine specific restrictions.
        """

        ParameterisedGains.restrict_solution(self)
        
        if self.ref_ant is not None:
            self.slope_params -= self.slope_params[:,:,:,self.ref_ant,:,:,:][:,:,:,np.newaxis,:,:,:]
        for idir in self.fix_directions:
            self.slope_params[idir, ...] = 0

    def precompute_attributes(self, model_arr, flags_arr, inv_var_chan):
        """
        Precompute (J\ :sup:`H`\J)\ :sup:`-1`, which does not vary with iteration.

        Args:
            model_arr (np.ndarray):
                Shape (n_dir, n_mod, n_tim, n_fre, n_ant, n_ant, n_cor, n_cor) array containing 
                model visibilities.
        """
        ParameterisedGains.precompute_attributes(self, model_arr, flags_arr, inv_var_chan)

        jhj1_shape = [self.n_dir, self.n_tim, self.n_fre, self.n_ant, 2, 2]

        jhj1 = self.cyslope.allocate_gain_array(jhj1_shape, dtype=self.dtype, zeros=True)

        # use appropriate phase-only kernel (with 1,1 intervals) to compute inner JHJ
        self.cykernel.cycompute_jhj(model_arr, jhj1, 1,1)

        blocks_per_inverse = 6 if self.slope_type=="tf-plane" else 3

        jhj_shape = [self.n_dir, self.n_timint, self.n_freint, self.n_ant, blocks_per_inverse, 2, 2]

        jhj = self.cyslope.allocate_param_array(jhj_shape, dtype=self.ftype, zeros=True)

        self.cyslope.cycompute_jhj(jhj1.real, jhj, self.chunk_ts, self.chunk_fs, self.t_int, self.f_int)

        self.jhjinv = np.zeros_like(jhj)

        self.cyslope.cycompute_jhjinv(jhj, self.jhjinv, self.eps)
