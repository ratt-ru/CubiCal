# CubiCal: a radio interferometric calibration suite
# (c) 2017 Rhodes University & Jonathan S. Kenyon
# http://github.com/ratt-ru/CubiCal
# This code is distributed under the terms of GPLv2, see LICENSE.md for details
from __future__ import print_function
from cubical.machines.interval_gain_machine import PerIntervalGains
import numpy as np
from cubical.flagging import FL
import cubical.kernels
from numpy.ma import masked_array

from .abstract_machine import log

class PhaseDiagGains(PerIntervalGains):
    """
    This class implements the diagonal phase-only gain machine.
    """
    def __init__(self, label, data_arr, ndir, nmod, chunk_ts, chunk_fs, chunk_label, options):
        """
        Initialises a diagonal phase-only gain machine.
        
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
        PerIntervalGains.__init__(self, label, data_arr, ndir, nmod,
                                  chunk_ts, chunk_fs, chunk_label, options)

        # kernel used in solver is diag-diag in diag mode, else uses full kernel version
        if options.get('diag-data') or options.get('diag-only'):
            self.kernel_solve = cubical.kernels.import_kernel('diag_phase_only')
        else:
            self.kernel_solve = cubical.kernels.import_kernel('phase_only')

        self.phases = self.kernel.allocate_gain_array(self.gain_shape, dtype=self.ftype, zeros=True)
        self.gains = np.empty_like(self.phases, dtype=self.dtype)
        self.gains[:] = np.eye(self.n_cor) 
        self.old_gains = self.gains.copy()

    @classmethod
    def determine_diagonality(cls, options):
        return True

    @classmethod
    def get_full_kernel(cls, options, diag_gains):
        if options.get('diag-data'):
            return cubical.kernels.import_kernel('diag_phase_only')
        else:
            return cubical.kernels.import_kernel('phase_only')

    def get_inverse_gains(self):
        """Returns inverse gains and inverse conjugate gains. For phase-only, conjugation is inverse"""
        gh = self.get_conj_gains()
        return gh, self.gains, 0

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

        gh = self.get_conj_gains()
        jh = self.get_new_jh(model_arr)

        self.kernel_solve.compute_jh(model_arr, self.gains, jh, self.t_int, self.f_int)

        jhr = self.get_new_jhr()

        r = self.get_obs_or_res(obser_arr, model_arr)

        self.kernel_solve.compute_jhr(gh, jh, r, jhr, self.t_int, self.f_int)

        return jhr.imag, self.jhjinv, 0


    @property
    def dof_per_antenna(self):
        """This property returns the number of real degrees of freedom per antenna, per solution interval"""
        return 2

    @staticmethod
    def exportable_solutions():
        """ Returns a dictionary of exportable solutions for this machine type. """

        exportables = PerIntervalGains.exportable_solutions()

        exportables.update({
            "phase": (0., ("dir", "time", "freq", "ant", "corr")),
            "phase.err": (0., ("dir", "time", "freq", "ant", "corr")),
        })

        return exportables

    def importable_solutions(self, grid0):
        """ Returns a dictionary of importable solutions for this machine type. """
        # can import 2x2 complex gain, with a corr1/corr2 axis
        solutions = PerIntervalGains.importable_solutions(self, grid0)

        # but also phase (with a corr axis)
        solutions['phase'] = dict(dir=grid0['dir'] if self.dd_term else [0], ant=grid0['ant'], corr=grid0['corr'], **self.interval_grid)

        return solutions

    def export_solutions(self):
        """ Saves the solutions to a dict of {label: solutions,grids} items. """

        solutions = super(PhaseDiagGains, self).export_solutions()

        # construct phase solutions, applying mask from parent gains
        mask = solutions['gain'][0].mask
        solutions['phase'] = masked_array(self.phases, mask)[..., (0, 1), (0, 1)], self.interval_grid

        # phase error is same as gain error (small angle approx!)
        solutions["phase.err"] = solutions["gain.err"][0][..., (0, 1), (0, 1)], self.interval_grid

        return solutions

    def import_solutions(self, soldict):
        """
        Loads solutions from a dict.

        Args:
            soldict (dict):
                Contains gains solutions which must be loaded.
        """
        # load from phase term, if present
        if "phase" in soldict:
            self.phases[...,(0,1),(0,1)] = soldict["phase"].data
            print("loading phases directly", file=log(0))

        # else try to load from gain term
        elif "gain" in soldict:
            self.phases[..., (0,1), (0,1)] = np.angle(soldict["gain"].data)[..., (0,1), (0,1)]
            self.phases[...,0,1].fill(0)
            self.phases[...,1,0].fill(0)
            print("loading phase component from gain", file=log(0))
        else:
            return

        # loaded -- do the housekeeping

        self.restrict_solution(self.phases)
        np.multiply(self.phases, 1j, out=self.gains)
        np.exp(self.gains, out=self.gains)
        self._gains_loaded = True


    def implement_update(self, jhr, jhjinv):

        # variance of gain is diagonal of jhjinv
        if self.posterior_gain_error is None:
            self.posterior_gain_error = np.sqrt(jhjinv.real)
        else:
            np.sqrt(jhjinv.real, out=self.posterior_gain_error)

        update = self.init_update(jhr)

        self.kernel_solve.compute_update(jhr, jhjinv, update)

        if self.iters%2 == 0:
            update *= 0.5
        self.phases += update

        self.restrict_solution(self.phases)

        np.multiply(self.phases, 1j, out=self.gains)
        np.exp(self.gains, out=self.gains)
        self.gains[...,0,1].fill(0)
        self.gains[...,1,0].fill(0)

        # raise flag so updates of G^H and G^-1 are computed
        self._gh_update = self._ghinv_update = True

    def restrict_solution(self, phases):
        """
        Restricts the solution by invoking the inherited restrict_soultion method and applying
        any machine specific restrictions.
        """

        # PerIntervalGains.restrict_solution(self)

        if self.ref_ant is not None:
            phases -= phases[:,:,:,self.ref_ant,0,0][:,:,:,np.newaxis,np.newaxis,np.newaxis]
        for idir in self.fix_directions:
            phases[idir, ...] = 0


    def precompute_attributes(self, data_arr, model_arr, flags_arr, noise):
        """
        Precompute (J\ :sup:`H`\J)\ :sup:`-1`, which does not vary with iteration.

        Args:
            model_arr (np.ndarray):
                Shape (n_dir, n_mod, n_tim, n_fre, n_ant, n_ant, n_cor, n_cor) array containing 
                model visibilities.
        """
        PerIntervalGains.precompute_attributes(self, data_arr, model_arr, flags_arr, noise)

        self.jhjinv = np.zeros_like(self.gains)

        self.kernel_solve.compute_jhj(model_arr, self.jhjinv, self.t_int, self.f_int)

        self.kernel_solve.compute_jhjinv(self.jhjinv, self.jhjinv, self.gflags, self.eps, FL.ILLCOND)

        self.jhjinv = self.jhjinv.real

