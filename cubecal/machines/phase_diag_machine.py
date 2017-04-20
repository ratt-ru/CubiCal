from abstract_gain_machine import PerIntervalGains
import numpy as np
import cyfull_complex as cyfull

class PhaseDiagGains(PerIntervalGains):
    """
    This class implements the diagonal phase-only gain machine.
    """
    def __init__(self, model_arr, options):
        PerIntervalGains.__init__(self, model_arr, options)
        self.gains     = np.empty(self.gain_shape, dtype=self.dtype)
        self.gains[:]  = np.eye(self.n_cor)
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

	    float_type = np.float64 if obser_arr.dtype is np.complex128 else np.float32

	    gh = self.gains.transpose(0,1,2,3,5,4).conj()

	    jh = np.zeros_like(model_arr)

	    cyphase.cycompute_jh(model_arr, self.gains, jh, self.t_int, self.f_int)

	    jhr_shape = [n_dir, n_timint, n_freint, n_ant, n_cor, n_cor]

	    jhr = np.zeros(jhr_shape, dtype=obser_arr.dtype)

	    # TODO: This breaks with the new compute residual code for n_dir > 1. Will need a fix.
	    if n_dir > 1:
	        r = compute_residual(obser_arr, model_arr, gains, t_int, f_int)
	    else:
	        r = obser_arr

	    cyphase.cycompute_jhr(gh, jh, r, jhr, t_int, f_int)

	    return np.imag(jhr)