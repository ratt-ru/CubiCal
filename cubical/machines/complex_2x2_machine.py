# CubiCal: a radio interferometric calibration suite
# (c) 2017 Rhodes University & Jonathan S. Kenyon
# http://github.com/ratt-ru/CubiCal
# This code is distributed under the terms of GPLv2, see LICENSE.md for details
from __future__ import print_function
from cubical.machines.interval_gain_machine import PerIntervalGains
import numpy as np
import numpy.ma as ma
from numpy.ma import masked_array
from cubical.flagging import FL
import cubical.kernels

from cubical.tools import logger
log = logger.getLogger("complex_2x2")


class Complex2x2Gains(PerIntervalGains):
    """
    This class implements the full complex 2x2 gain machine.
    """
    def __init__(self, label, data_arr, ndir, nmod, chunk_ts, chunk_fs, chunk_label, options):
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
            nmod (nmod):
                Number of models.
            chunk_ts (np.ndarray):
                Times for the data being processed.
            chunk_fs (np.ndarray):
                Frequencies for the data being processsed.
            options (dict): 
                Dictionary of options. 
        """
        # note that this sets up self.kernel
        PerIntervalGains.__init__(self, label, data_arr, ndir, nmod,
                                  chunk_ts, chunk_fs, chunk_label, options)

        # only solve for off-diagonal terms
        self._offdiag_only = options["offdiag-only"]

#        if label == "D":
#            self.gains[:,:,:,:,1,1] = -1

    @classmethod
    def determine_diagonality(cls, options):
        """Returns true if the machine class, given the options, represents a diagonal gain"""
        return options['type'] == 'complex-diag' or options['update-type'] != 'full'

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

        self.kernel_solve.compute_jh(model_arr, self.gains, jh, self.t_int, self.f_int)

        jhr = self.get_new_jhr()
        r = self.get_obs_or_res(obser_arr, model_arr)

        if self._offdiag_only:
            jh[...,(0,1),(0,1)] = 0
            r[...,(0,1),(0,1)] = 0

        self.kernel_solve.compute_jhr(jh, r, jhr, self.t_int, self.f_int)

        jhj, jhjinv = self.get_new_jhj()

        self.kernel_solve.compute_jhj(jh, jhj, self.t_int, self.f_int)

        if "scalar" in self.update_type:
            jhr[..., (0, 1), (0, 1)] = jhr[..., (0, 1), (0, 1)].mean(axis=-1)[..., np.newaxis]
            jhj[..., (0, 1), (0, 1)] = jhj[..., (0, 1), (0, 1)].mean(axis=-1)[..., np.newaxis]

        flag_count = self.kernel_solve.compute_jhjinv(jhj, jhjinv, self.gflags, self.eps, FL.ILLCOND)

#         if flag_count:
#             import pdb; pdb.set_trace()

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

        self.kernel_solve.compute_update(jhr, jhjinv, update)

        if self.dd_term and self.n_dir > 1:
            update += self.gains

        self.restrict_solution(update)

        #if self._offdiag_only:
        #    update[...,0,0] = 1
        #    update[:,:,:,:,1,1] = self._exp_pzd[np.newaxis,:,:,np.newaxis]

        if self.iters % 2 == 0 or self.n_dir > 1:
            self.gains += update
            self.gains *= 0.5
            self.restrict_solution(self.gains)
        else:
            np.copyto(self.gains, update)

    def restrict_solution(self, gains):
        """
        Restricts the solution by invoking the inherited restrict_soultion method and applying
        any machine specific restrictions.
        """
        if self.ref_ant is not None:
            phase = np.angle(self.gains[...,self.ref_ant,0,0])
            gains[:,:,:,:,(0,1),(0,1)] *= np.exp(-1j*phase)[:,:,:,np.newaxis,np.newaxis]

        super(Complex2x2Gains, self).restrict_solution(gains)

