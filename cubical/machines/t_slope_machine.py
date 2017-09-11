from cubical.machines.parameterised_machine import ParameterisedGains
import numpy as np
import cubical.kernels.cyt_slope as cyslope
from cubical.flagging import FL

class PhaseSlopeGains(ParameterisedGains):
    """
    This class implements the diagonal phase-only gain machine.
    """
    def __init__(self, label, data_arr, ndir, nmod, chunk_ts, chunk_fs, options):
        
        ParameterisedGains.__init__(self, label, data_arr, ndir, nmod, chunk_ts, chunk_fs, options)

        self.param_shape = [self.n_dir, self.n_timint, self.n_freint, self.n_ant, 2, self.n_cor, self.n_cor]
        self.slope_params = np.zeros(self.param_shape, dtype=self.ftype)

        self.gains = np.empty(self.gain_shape, dtype=self.dtype)
        self.gains[:] = np.eye(self.n_cor) 
        self.old_gains = self.gains.copy()

        self.chunk_ts = (chunk_ts - chunk_ts[0])/(chunk_ts[-1] - chunk_ts[0])

    def compute_js(self, obser_arr, model_arr):
        """
        This method computes the (J^H)R term of the GN/LM method for the
        full-polarisation, diagonal phase-only gains case.

        Args:
            obser_arr (np.array): Array containing the observed visibilities.
            model_arr (np.array): Array containing the model visibilities.

        Returns:
            jhr (np.array): Array containing the result of computing (J^H)R.
        """

        n_dir, n_tim, n_fre, n_ant, n_cor, n_cor = self.gains.shape

        gh = self.gains.transpose(0,1,2,3,5,4).conj()

        jh = np.zeros_like(model_arr)

        cyslope.cycompute_jh(model_arr, self.gains, jh, 1, 1)

        tmp_jhr_shape = [n_dir, n_tim, n_fre, n_ant, n_cor, n_cor]

        tmp_jhr = np.zeros(tmp_jhr_shape, dtype=obser_arr.dtype)

        if n_dir > 1:
            resid_arr = np.empty_like(obser_arr)
            r = self.compute_residual(obser_arr, model_arr, resid_arr)
        else:
            r = obser_arr

        cyslope.cycompute_tmp_jhr(gh, jh, r, tmp_jhr, 1, 1)

        tmp_jhr = tmp_jhr.imag

        jhr_shape = [n_dir, self.n_timint, self.n_freint, n_ant, 2, n_cor, n_cor]

        jhr = np.zeros(jhr_shape, dtype=tmp_jhr.dtype)

        cyslope.cycompute_jhr(tmp_jhr, jhr, self.chunk_ts, self.t_int, self.f_int)

        return jhr

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

        jhr = self.compute_js(obser_arr, model_arr)

        update = np.zeros_like(jhr)

        cyslope.cycompute_update(jhr, self.jhjinv, update)

        if self.iters%2 == 0:
            self.slope_params += 0.5*update
        else:
            self.slope_params += update

        self.restrict_solution()

        # Need to turn updated parameters into gains.

        cyslope.cyconstruct_gains(self.slope_params, self.gains, self.chunk_ts, self.t_int, self.f_int)

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

        resid_arr[:] = obser_arr

        cyslope.cycompute_residual(model_arr, self.gains, gains_h, resid_arr, 1, 1)

        return resid_arr

    def apply_inv_gains(self, resid_arr, corr_vis=None):
        """
        Applies the inverse of the gain estimates to the observed data matrix.

        Args:
            obser_arr (np.array): Array of the observed visibilities.
            gains (np.array): Array of the gain estimates.

        Returns:
            inv_gdgh (np.array): Array containing (G^-1)D(G^-H).
        """

        g_inv = self.gains.conj()

        gh_inv = g_inv.conj()

        if corr_vis is None:                
            corr_vis = np.empty_like(resid_arr)

        cyslope.cycompute_corrected(resid_arr, g_inv, gh_inv, corr_vis, 1, 1)

        return corr_vis, 0   # no flags raised here, since phase-only always invertible

    def apply_gains(self):
        """
        This method should be able to apply the gains to an array at full time-frequency
        resolution. Should return the input array at full resolution after the application of the 
        gains.
        """
        return

    def restrict_solution(self):

        ParameterisedGains.restrict_solution(self)
        
        if self.ref_ant is not None:
            self.slope_params -= self.slope_params[:,:,:,self.ref_ant,:,:,:][:,:,:,np.newaxis,:,:,:]

    def precompute_attributes(self, model_arr):

        tmp_jhj_shape = [self.n_dir, self.n_mod, self.n_tim, self.n_fre, self.n_ant, 2, 2] 

        tmp_jhj = np.zeros(tmp_jhj_shape, dtype=self.dtype)

        cyslope.cycompute_tmp_jhj(model_arr, tmp_jhj)

        jhj_shape = [self.n_dir, self.n_timint, self.n_freint, self.n_ant, 3, 2, 2]

        jhj = np.zeros(jhj_shape, dtype=self.ftype)

        cyslope.cycompute_jhj(tmp_jhj.real, jhj, self.chunk_ts, self.t_int, self.f_int)

        self.jhjinv = np.zeros(jhj_shape, dtype=self.ftype)

        cyslope.cycompute_jhjinv(jhj, self.jhjinv, self.eps)
