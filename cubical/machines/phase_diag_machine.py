# CubiCal: a radio interferometric calibration suite
# (c) 2017 Rhodes University & Jonathan S. Kenyon
# http://github.com/ratt-ru/CubiCal
# This code is distributed under the terms of GPLv2, see LICENSE.md for details
from cubical.machines.interval_gain_machine import PerIntervalGains
import numpy as np
from cubical.flagging import FL
import cubical.kernels

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
                                  chunk_ts, chunk_fs, chunk_label, options,
                                  self.get_kernel(options))

        self.phases = cykernel.allocate_gain_array(self.gain_shape, dtype=self.ftype, zeros=True)
        self.gains = np.empty_like(self.phases, dtype=self.dtype)
        self.gains[:] = np.eye(self.n_cor) 
        self.old_gains = self.gains.copy()

    @staticmethod
    def get_kernel(options):
        """Returns kernel approriate to Jones options"""
        if options['diag-diag']:
            return cubical.kernels.import_kernel('cydiag_phase_only')
        else:
            return cubical.kernels.import_kernel('cyphase_only')

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

        n_dir, n_timint, n_freint, n_ant, n_cor, n_cor = self.gains.shape

        gh = self.get_conj_gains()
        jh = self.get_new_jh(model_arr)

        self.cykernel.cycompute_jh(model_arr, self.gains, jh, self.t_int, self.f_int)

        jhr = self.get_new_jhr()

        r = self.get_obs_or_res(obser_arr, model_arr)

        self.cykernel.cycompute_jhr(gh, jh, r, jhr, self.t_int, self.f_int)

        return jhr.imag, self.jhjinv, 0


    @property
    def dof_per_antenna(self):
        """This property returns the number of real degrees of freedom per antenna, per solution interval"""
        return 2

    def implement_update(self, jhr, jhjinv):

        # variance of gain is diagonal of jhjinv
        if self.posterior_gain_error is None:
            self.posterior_gain_error = np.sqrt(jhjinv.real)
        else:
            np.sqrt(jhjinv.real, out=self.posterior_gain_error)

        update = self.init_update(jhr)

        self.cykernel.cycompute_update(jhr, jhjinv, update)

        if self.iters%2 == 0:
            update *= 0.5
        self.phases += update

        self.restrict_solution()

        np.multiply(self.phases, 1j, out=self.gains)
        np.exp(self.gains, out=self.gains)
        self.gains[...,0,1].fill(0)
        self.gains[...,1,0].fill(0)

    def restrict_solution(self):
        """
        Restricts the solution by invoking the inherited restrict_soultion method and applying
        any machine specific restrictions.
        """

        PerIntervalGains.restrict_solution(self)

        if self.ref_ant is not None:
            self.phases -= self.phases[:,:,:,self.ref_ant,:,:][:,:,:,np.newaxis,:,:]
        for idir in self.fix_directions:
            self.phases[idir, ...] = 0


    def precompute_attributes(self, model_arr, flags_arr, noise):
        """
        Precompute (J\ :sup:`H`\J)\ :sup:`-1`, which does not vary with iteration.

        Args:
            model_arr (np.ndarray):
                Shape (n_dir, n_mod, n_tim, n_fre, n_ant, n_ant, n_cor, n_cor) array containing 
                model visibilities.
        """
        PerIntervalGains.precompute_attributes(self, model_arr, flags_arr, noise)

        self.jhjinv = np.zeros_like(self.gains)

        self.cykernel.cycompute_jhj(model_arr, self.jhjinv, self.t_int, self.f_int)

        self.cykernel.cycompute_jhjinv(self.jhjinv, self.jhjinv, self.gflags, self.eps, FL.ILLCOND)

        self.jhjinv = self.jhjinv.real

