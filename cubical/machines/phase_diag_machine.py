from cubical.machines.interval_gain_machine import PerIntervalGains
import numpy as np
import cubical.kernels.cyphase_only as cyphase

class PhaseDiagGains(PerIntervalGains):
    """
    This class implements the diagonal phase-only gain machine.
    """
    def __init__(self, model_arr, chunk_ts, chunk_fs, options):
        
        PerIntervalGains.__init__(self, model_arr, chunk_ts, chunk_fs, options)

        self.float_type = np.float64 if model_arr.dtype is np.complex128 else np.float32

        self.phases = np.zeros(self.gain_shape, dtype=self.float_type)

        self.gains = np.empty_like(self.phases, dtype=model_arr.dtype)
        self.gains[:] = np.eye(self.n_cor) 
        self.old_gains = self.gains.copy()

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

        n_dir, n_timint, n_freint, n_ant, n_cor, n_cor = self.gains.shape

        gh = self.gains.transpose(0,1,2,3,5,4).conj()

        jh = np.zeros_like(model_arr)

        cyphase.cycompute_jh(model_arr, self.gains, jh, self.t_int, self.f_int)

        jhr_shape = [n_dir, n_timint, n_freint, n_ant, n_cor, n_cor]

        jhr = np.zeros(jhr_shape, dtype=obser_arr.dtype)

        # TODO: This breaks with the new compute residual code for n_dir > 1. Will need a fix.
        if n_dir > 1:
            resid_arr = np.empty_like(obser_arr)
            r = self.compute_residual(obser_arr, model_arr, resid_arr)
        else:
            r = obser_arr

        cyphase.cycompute_jhr(gh, jh, r, jhr, self.t_int, self.f_int)

        return jhr.imag

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
            
        self.iters = self.iters + 1

        jhr = self.compute_js(obser_arr, model_arr)

        update = np.zeros_like(jhr)

        cyphase.cycompute_update(jhr, self.jhjinv, update)

        if self.iters%2 == 0:
            self.phases += 0.5*update
        else:
            self.phases += update

        self.phases = self.phases - self.phases[:,:,:,0:1,:,:]

        self.gains = np.exp(1j*self.phases)
        self.gains[...,(0,1),(1,0)] = 0 

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

        cyphase.cycompute_residual(model_arr, self.gains, gains_h, resid_arr, self.t_int, self.f_int)

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

        g_inv = self.gains.conj()

        gh_inv = g_inv.conj()

        if corr_vis is None:                
            corr_vis = np.empty_like(obser_arr)

        cyphase.cycompute_corrected(obser_arr, g_inv, gh_inv, corr_vis, self.t_int, self.f_int)

        return corr_vis, 0   # no flags raised here, since phase-only always invertible

    def apply_gains(self):
        """
        This method should be able to apply the gains to an array at full time-frequency
        resolution. Should return the input array at full resolution after the application of the 
        gains.
        """
        return

    def precompute_attributes(self, model_arr):

        self.jhjinv = np.zeros_like(self.gains)

        cyphase.cycompute_jhj(model_arr, self.jhjinv, self.t_int, self.f_int)

        cyphase.cycompute_jhjinv(self.jhjinv, self.gflags, self.eps, self.flagbit)

        self.jhjinv = self.jhjinv.real

