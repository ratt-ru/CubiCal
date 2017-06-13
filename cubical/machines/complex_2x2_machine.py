from cubical.machines.interval_gain_machine import PerIntervalGains
import numpy as np
import cubical.kernels.cyfull_complex as cyfull

class Complex2x2Gains(PerIntervalGains):
    """
    This class implements the full complex 2x2 gain machine
    """
    def __init__(self, model_arr, chunk_ts, chunk_fs, options):
        PerIntervalGains.__init__(self, model_arr, chunk_ts, chunk_fs, options)
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
            jhjinv (np.array): Array containing the result of computing (J^H.J)^-1
            flag_count:     Number of flagged (ill-conditioned) elements
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
                (((J^H)J)^-1)(J^H)R
        """


        jhr, jhjinv, flag_count = self.compute_js(obser_arr, model_arr)

        update = np.empty_like(jhr)

        cyfull.cycompute_update(jhr, jhjinv, update)

        if model_arr.shape[0]>1:
            update = self.gains + update

        if iters % 2 == 0:
            self.gains = 0.5*(self.gains + update)
        else:
            self.gains = update

        return flag_count


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

        cyfull.cycompute_residual(model_arr, self.gains, gains_h, resid_arr, self.t_int, self.f_int)

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

        flag_count = cyfull.cycompute_jhjinv(self.gains, g_inv, self.gflags, self.eps, self.flagbit) # Function can invert G.

        gh_inv = g_inv.transpose(0,1,2,3,5,4).conj()

        if corr_vis is None:
            corr_vis = np.empty_like(obser_arr)

        cyfull.cycompute_corrected(obser_arr, g_inv, gh_inv, corr_vis, self.t_int, self.f_int)

        return corr_vis, flag_count
         
    def apply_gains(self):
        """
        This method should be able to apply the gains to an array at full time-frequency
        resolution. Should return the input array at full resolution after the application of the 
        gains.
        """
        return
          
    # def compute_stats(self):
    #     """
    #     This method should compute a variety of useful parameters regarding the conditioning and 
    #     degrees of freedom of the current time-frequency chunk. Specifically, it must populate 
    #     an attribute containing the degrees of freedom per time-frequency slot. 
    #     """
    #     return
          
    def is_converged(self):
        """
        This method should check the convergence of the current time-frequency chunk. Should return 
        a Boolean.
        """
        return
          
    def compute_chi_squared(self):
        """
        Ignore for now - will likey form part of the Jones Chain.
        """
        return