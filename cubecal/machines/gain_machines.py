"""
Implements the full complex 2x2 GainMachine
"""

import cyfull_complex as cyfull
from Tools import logger
import numpy as np
import math

log = logger.getLogger("full_complex")


class PerIntervalGains(object):
    """
    This is a base class for all gain solution machines that use solutions intervals.
    """
    def __init__(self, model_arr, options):
        """Given a model array shape, initializes various sizes relevant to gain solutions"""
        self.n_dir, self.n_tim, self.n_fre, self.n_ant, self.n_ant, self.n_cor, self.n_cor = model_arr.shape
        self.dtype = model_arr.dtype
        self.t_int = options["time-int"]
        self.f_int = options["freq-int"]

        # n_tim and n_fre are the time and frequency dimensions of the data arrays.
        # n_timint and n_freint are the time and frequnecy dimensions of the gains.

        self.n_timint = int(math.ceil(float(self.n_tim) / self.t_int))  # Number of time intervals
        self.n_freint = int(math.ceil(float(self.n_fre) / self.f_int))  # Number of freq intervals
        self.n_tf = self.n_fre * self.n_tim         # Number of time-freq slots
        self.n_int = self.n_timint * self.n_freint  # Total number of solution intervals

        # total number of solutions
        self.n_sols = float(self.n_dir * self.n_int)

        # Initialize gains to the appropriate shape with all gains set to identity. Create a copy to
        # hold the gain of the previous iteration.

        self.gain_shape = [self.n_dir, self.n_timint, self.n_freint, self.n_ant, self.n_cor, self.n_cor]


class Complex2x2Gains(PerIntervalGains):
    """
    This class implements the full complex 2x2 gain machine
    """
    def __init__(self, model_arr, options):
        PerIntervalGains.__init__(self, model_arr, options)
        self.gains     = np.empty(self.gain_shape, dtype=self.dtype)
        self.gains[:]  = np.eye(self.n_cor)
        self.old_gains = self.gains.copy()

    def compute_js(self, obser_arr, model_arr):
        """
        This function computes the (J^H)R term of the GN/LM method for the
        full-polarisation, phase-only case.

        Args:
            obser_arr (np.array): Array containing the observed visibilities.
            model_arr (np.array): Array containing the model visibilities.
            gains (np.array): Array containing the current gain estimates.

        Returns:
            jhr (np.array): Array containing the result of computing (J^H)R.
        """

        n_dir, n_tim, n_fre, n_ant, n_cor, n_cor = self.gains.shape

        jh = np.zeros_like(model_arr)

        cyfull.cycompute_jh(model_arr, self.gains, jh, self.t_int, self.f_int)

        jhr_shape = [n_dir, n_tim, n_fre, n_ant, n_cor, n_cor]

        jhr = np.zeros(jhr_shape, dtype=obser_arr.dtype)

        # TODO: This breaks with the new compute residual code for n_dir > 1. Will need a fix.
        if n_dir > 1:
            r = self.compute_residual(obser_arr, model_arr)
        else:
            r = obser_arr

        cyfull.cycompute_jhr(jh, r, jhr, self.t_int, self.f_int)

        jhj = np.zeros(jhr_shape, dtype=obser_arr.dtype)

        cyfull.cycompute_jhj(jh, jhj, self.t_int, self.f_int)

        jhjinv = np.empty(jhr_shape, dtype=obser_arr.dtype)

        cyfull.cycompute_jhjinv(jhj, jhjinv)

        return jhr, jhjinv

    def compute_update(self, model_arr, obser_arr):
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
                (((J^H)J)^-1)(J^H)R
        """


        jhr, jhjinv = self.compute_js(obser_arr, model_arr)

        update = np.empty_like(jhr)

        cyfull.cycompute_update(jhr, jhjinv, update)

        return update


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


    def apply_inv_gains(self, obser_arr):
        """
        Applies the inverse of the gain estimates to the observed data matrix.

        Args:
            obser_arr (np.array): Array of the observed visibilities.
            gains (np.array): Array of the gain estimates.

        Returns:
            inv_gdgh (np.array): Array containing (G^-1)D(G^-H).
        """

        g_inv = np.empty_like(self.gains)

        cyfull.cycompute_jhjinv(self.gains, g_inv) # Function can invert G.

        gh_inv = g_inv.transpose(0,1,2,3,5,4).conj()

        corr_vis = np.empty_like(obser_arr)

        cyfull.cycompute_corrected(obser_arr, g_inv, gh_inv, corr_vis, self.t_int, self.f_int)

        return corr_vis

