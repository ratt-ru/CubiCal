# CubiCal: a radio interferometric calibration suite
# (c) 2017 Rhodes University & Jonathan S. Kenyon
# http://github.com/ratt-ru/CubiCal
# This code is distributed under the terms of GPLv2, see LICENSE.md for details
from cubical.machines.interval_gain_machine import PerIntervalGains
import numpy as np
import cubical.kernels.cyfull_W_complex as cyfull

class ComplexW2x2Gains(PerIntervalGains):
    """
    This class implements the full complex 2x2 gain machine
    """
    def __init__(self, model_arr, options):
        PerIntervalGains.__init__(self, model_arr, options)
        self.gains     = np.empty(self.gain_shape, dtype=self.dtype)
        self.gains[:]  = np.eye(self.n_cor) #np.ones(self.ncor)
        self.weights_shape = [self.n_mod, self.n_tim, self.n_fre, self.n_ant, self.n_ant]
        self.weights = np.ones(self.weights_shape, dtype=np.float64)
        self.v_shape = [self.n_mod, self.n_timint, self.n_freint]
        self.v = 2*np.ones(self.v_shape, dtype=np.float64)

    def compute_js(self, obser_arr, model_arr):
        """
        This function computes the (J^H)WR term of the GN/LM method for the
        full-polarisation, phase-only case.

        Args:
            obser_arr (np.array): Array containing the observed visibilities.
            model_arr (np.array): Array containing the model visibilities.
            gains (np.array): Array containing the current gain estimates.

        Returns:
            jhwr (np.array): Array containing the result of computing (J^H)R.
        """
        w = self.weights
        n_dir, n_tim, n_fre, n_ant, n_cor, n_cor = self.gains.shape

        jh = np.zeros_like(model_arr)

        cyfull.cycompute_jh(model_arr, self.gains, jh, self.t_int, self.f_int)

        jhwr_shape = [n_dir, n_tim, n_fre, n_ant, n_cor, n_cor]

        jhwr = np.zeros(jhwr_shape, dtype=obser_arr.dtype)

        # TODO: This breaks with the new compute residual code for n_dir > 1. Will need a fix.
        if n_dir > 1:
            r = self.compute_residual(obser_arr, model_arr)
        else:
            r = obser_arr

        cyfull.cycompute_jhwr(jh, r, w, jhwr, self.t_int, self.f_int)

        jhwj = np.zeros(jhwr_shape, dtype=obser_arr.dtype)

        cyfull.cycompute_jhwj(jh, w, jhwj, self.t_int, self.f_int)

        jhwjinv = np.empty(jhwr_shape, dtype=obser_arr.dtype)

        cyfull.cycompute_jhwjinv(jhwj, jhwjinv)

        return jhwr, jhwjinv

    def compute_update(self, model_arr, obser_arr, iters):
        """
        This function computes the update step of the GN/LM method. This is
        equivalent to the complete (((J^H)J)^-1)(J^H)R.

        Args:
            obser_arr (np.array): Array containing the observed visibilities.
            model_arr (np.array): Array containing the model visibilities.
            gains (np.array): Array containing the current gain estimates.
            jhjinv (np.array): Array containing (J^H)J)^-1. (Invariant)

        Returns:
            update (np.array): Array containing the result of computing
                (((J^H)WJ)^-1)(J^H)WR
        """


        jhwr, jhwjinv = self.compute_js(obser_arr, model_arr)

        update = np.empty_like(jhwr)

        cyfull.cycompute_update(jhwr, jhwjinv, update)

        if iters % 2 == 0:
            self.gains = 0.5*(self.gains + update)
        else:
            self.gains = update

        #Computing the weights
        res_shape = [self.n_mod, self.n_tim, self.n_fre, self.n_ant, self.n_ant, self.n_cor, self.n_cor]
        
        residuals = np.zeros(res_shape, dtype=self.dtype) #We need to always compute the residuals
        
        residuals = self.compute_residual(obser_arr, model_arr, residuals)
        
        weights, v = cyfull.cycompute_weights(residuals, self.weights, self.v, self.t_int, self.f_int)

        self.weights = weights

        self.v = v



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

        gains_h = self.gains.transpose(0,1,2,3,5,4).conj()

        cyfull.cycompute_residual(model_arr, self.gains, gains_h, obser_arr, resid_arr, self.t_int, self.f_int)

        return resid_arr


    def apply_inv_gains(self, obser_arr, corr_vis=None):
        """
        Applies the inverse of the gain estimates to the observed data matrix.

        Args:
            obser_arr (np.array): Array of the observed visibilities.
            gains (np.array): Array of the gain estimates.

        Returns:
            inv_gdgh (np.array): Array containing (G^-1)D(G^-H).
        """

        g_inv = np.empty_like(self.gains)

        cyfull.cycompute_jhwjinv(self.gains, g_inv) # Function can invert G.

        gh_inv = g_inv.transpose(0,1,2,3,5,4).conj()

        if corr_vis is None:
            corr_vis = np.empty_like(obser_arr)

        cyfull.cycompute_corrected(obser_arr, g_inv, gh_inv, corr_vis, self.t_int, self.f_int)

        return corr_vis
