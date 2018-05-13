# CubiCal: a radio interferometric calibration suite
# (c) 2017 Rhodes University & Jonathan S. Kenyon
# http://github.com/ratt-ru/CubiCal
# This code is distributed under the terms of GPLv2, see LICENSE.md for details
from cubical.machines.interval_gain_machine import PerIntervalGains
import numpy as np
from cubical.flagging import FL
import cubical.kernels

from cubical.tools import logger
log = logger.getLogger("complex_2x2")


class Complex2x2Gains(PerIntervalGains):
    """
    This class implements the full complex 2x2 gain machine.
    """
    def __init__(self, label, data_arr, ndir, nmod, double_precision, chunk_ts, chunk_fs, chunk_label, options):
        """
        Initialises a 2x2 complex gain machine.
        
        Args:
            label (str):
                Label identifying the Jones term.
            data_arr (np.ndarray): 
                Shape (n_mod, n_tim, n_fre, n_ant, n_ant, n_cor, n_cor) array containing observed 
                visibilities. 
            ndir (int):
                Number of directions.
            nmod (int):
                Number of models.
            double_precision (bool):
                Force use of double precision if True (else use dtype of data)
            chunk_ts (np.ndarray):
                Times for the data being processed.
            chunk_fs (np.ndarray):
                Frequencies for the data being processsed.
            options (dict): 
                Dictionary of options. 
        """
        PerIntervalGains.__init__(self, label, data_arr, ndir, nmod, double_precision,
                                  chunk_ts, chunk_fs, chunk_label, options,
                                  self.get_kernel(options))

    @staticmethod
    def get_kernel(options):
        """Returns kernel approriate to Jones options"""
        if options['diag-diag']:
            return cubical.kernels.import_kernel('cydiagdiag_complex')
        elif options['type'] == 'complex-2x2':
            return cubical.kernels.import_kernel('cyfull_complex')
        elif options['type'] == 'complex-diag':
            return cubical.kernels.import_kernel('cydiag_complex')
        else:
            raise RuntimeError("unknown machine type '{}'".format(options['type']))


    def compute_js(self, obser_arr, model_arr):
        """
        This function computes the (J\ :sup:`H`\J)\ :sup:`-1` and J\ :sup:`H`\R terms of the GN/LM 
        method. 

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

        n_dir, n_tim, n_fre, n_ant, n_cor, n_cor = self.gains.shape

        jh = self.get_new_jh(model_arr)

        self.cykernel.cycompute_jh(model_arr, self.gains, jh, self.t_int, self.f_int)

        jhr = self.get_new_jhr()
        r = self.get_obs_or_res(obser_arr, model_arr)

        self.cykernel.cycompute_jhr(jh, r, jhr, self.t_int, self.f_int)

        jhj, jhjinv = self.get_new_jhj()

        self.cykernel.cycompute_jhj(jh, jhj, self.t_int, self.f_int)

        flag_count = self.cykernel.cycompute_jhjinv(jhj, jhjinv, self.gflags, self.eps, FL.ILLCOND)

        return jhr, jhjinv, flag_count

    #@profile
    def implement_update(self, jhr, jhjinv):

        # jhjinv is 2x2 block-diagonal, with Hermitian blocks. TODO: what's the variance on the off-diagonals?
        # variance of gain is diagonal of jhjinv
        if self.posterior_gain_error is None:
            self.posterior_gain_error = np.zeros_like(jhjinv.real)
        diag = jhjinv[..., (0, 1), (0, 1)].real
        self.posterior_gain_error[...,(0,1),(0,1)] = np.sqrt(diag)
        self.posterior_gain_error[...,(1,0),(0,1)] = np.sqrt(diag.sum(axis=-1)/2)[...,np.newaxis]

        update = self.init_update(jhr)
        self.cykernel.cycompute_update(jhr, jhjinv, update)

        if self.dd_term and self.n_dir > 1:
            update += self.gains

        if self.iters % 2 == 0 or self.n_dir > 1:
            self.gains += update
            self.gains *= 0.5
        else:
            np.copyto(self.gains, update)

        self.restrict_solution()


    def restrict_solution(self):
        """
        Restricts the solution by invoking the inherited restrict_soultion method and applying
        any machine specific restrictions.
        """
        
        PerIntervalGains.restrict_solution(self)

        if self.ref_ant is not None:
            phase = np.angle(self.gains[...,self.ref_ant,(0,1),(0,1)])
            self.gains *= np.exp(-1j*phase)[:,:,:,np.newaxis,:,np.newaxis]