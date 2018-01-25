# CubiCal: a radio interferometric calibration suite
# (c) 2017 Rhodes University & Jonathan S. Kenyon
# http://github.com/ratt-ru/CubiCal
# This code is distributed under the terms of GPLv2, see LICENSE.md for details
from cubical.machines.interval_gain_machine import PerIntervalGains
import numpy as np
import cubical.kernels.cyfull_complex as cyfull

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
        PerIntervalGains.__init__(self, label, data_arr, ndir, nmod,
                                  chunk_ts, chunk_fs, chunk_label, options)
        
        self.gains     = np.empty(self.gain_shape, dtype=self.dtype)
        self.gains[:]  = np.eye(self.n_cor)
        self.old_gains = self.gains.copy()


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

        jh = np.zeros_like(model_arr)

        cyfull.cycompute_jh(model_arr, self.gains, jh, self.t_int, self.f_int)

        jhr_shape = [n_dir, n_tim, n_fre, n_ant, n_cor, n_cor]

        jhr = np.zeros(jhr_shape, dtype=obser_arr.dtype)

        # TODO: This breaks with the new compute residual code for n_dir > 1. Will need a fix.
        if n_dir > 1:
            resid_arr = np.empty_like(obser_arr)
            r = self.compute_residual(obser_arr, model_arr, resid_arr)
        else:
            r = obser_arr

        cyfull.cycompute_jhr(jh, r, jhr, self.t_int, self.f_int)

        jhj = np.zeros(jhr_shape, dtype=obser_arr.dtype)

        cyfull.cycompute_jhj(jh, jhj, self.t_int, self.f_int)

        jhjinv = np.empty(jhr_shape, dtype=obser_arr.dtype)

        flag_count = cyfull.cycompute_jhjinv(jhj, jhjinv, self.gflags, self.eps, self.flagbit)

        return jhr, jhjinv, flag_count

    def implement_update(self, jhr, jhjinv):
        update = np.empty_like(jhr)

        # variance of gain is diagonal of jhjinv
        self.posterior_gain_error = np.sqrt(jhjinv[...,(0,1),(0,1)].real)

        cyfull.cycompute_update(jhr, jhjinv, update)

        if self.dd_term and self.n_dir > 1:
            update = self.gains + update

        if self.iters % 2 == 0 or self.n_dir > 1:
            self.gains = 0.5*(self.gains + update)
        else:
            self.gains = update

        self.restrict_solution()


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

        gains_h = self.gains.transpose(0,1,2,3,5,4).conj()

        resid_arr[:] = obser_arr

        cyfull.cycompute_residual(model_arr, self.gains, gains_h, resid_arr, self.t_int, self.f_int)

        return resid_arr


    def restrict_solution(self):
        """
        Restricts the solution by invoking the inherited restrict_soultion method and applying
        any machine specific restrictions.
        """
        
        PerIntervalGains.restrict_solution(self)

        if self.ref_ant is not None:
            phase = np.angle(self.gains[...,self.ref_ant,(0,1),(0,1)])
            self.gains *= np.exp(-1j*phase)[:,:,:,np.newaxis,:,np.newaxis]