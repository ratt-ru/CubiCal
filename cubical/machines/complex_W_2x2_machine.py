from cubical.machines.interval_gain_machine import PerIntervalGains
import numpy as np
import cubical.kernels.cyfull_W_complex as cyfull
import cPickle
#from scipy.optimize import fsolve
from scipy import special
#import scipy.stats as st

class ComplexW2x2Gains(PerIntervalGains):
    """
    This class implements the weighted full complex 2x2 gain machine
    """
    
    def __init__(self, model_arr, chunk_ts, chunk_fs, label, options):
        PerIntervalGains.__init__(self, model_arr, chunk_ts, chunk_fs, options)
        
        self.gains     = np.empty(self.gain_shape, dtype=self.dtype)
        
        self.gains[:]  = np.eye(self.n_cor)
        
        self.old_gains = self.gains.copy()
        
        self.weights_shape = [self.n_mod, self.n_tim, self.n_fre, self.n_ant, self.n_ant, 1]
        
        self.weights = np.ones(self.weights_shape, dtype=self.dtype)
        
        self.v = 2.
        
        self.weight_dict = {}
        
        self.weight_dict["weights"] = {}
        
        self.weight_dict["vvals"] = {}
        
        self.label = label

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
            res_arr = np.empty_like(obser_arr)
            r = self.compute_residual(obser_arr, model_arr, res_arr)
        else:
            r = obser_arr

        cyfull.cycompute_jhwr(jh, r, w, jhwr, self.t_int, self.f_int)

        jhwj = np.zeros(jhwr_shape, dtype=obser_arr.dtype)

        cyfull.cycompute_jhwj(jh, w, jhwj, self.t_int, self.f_int)

        jhwjinv = np.empty(jhwr_shape, dtype=obser_arr.dtype)

        flag_count = cyfull.cycompute_jhwjinv(jhwj, jhwjinv, self.gflags, self.eps, self.flagbit)

        return jhwr, jhwjinv, flag_count

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

        
        jhwr, jhwjinv, flag_count = self.compute_js(obser_arr, model_arr)

        update = np.empty_like(jhwr)

        cyfull.cycompute_update(jhwr, jhwjinv, update)

        if model_arr.shape[0]>1:
            
            update = self.gains + update

        if iters % 2 == 0:
            
            self.gains = 0.5*(self.gains + update)
        
        else:
            
            self.gains = update

        #Computing the weights
        resid_arr = np.empty_like(obser_arr)
        residuals = self.compute_residual(obser_arr, model_arr, resid_arr)

    
        covinv = self.compute_covinv(residuals)

        weights = self.weights

        v = self.v
        
        self.weights, self.v = self.cycompute_weights(residuals, covinv, weights, v, self.t_int, self.f_int)

        self.weight_dict["weights"][iters] = self.weights
        
        self.weight_dict["vvals"][iters] = self.v
        
        self.weight_dict["t_int"] = self.t_int
        
        self.weight_dict["f_int"] = self.f_int
        
        f = open(str(self.label) + "_weights_dict.cp", "wb")
        
        cPickle.dump(self.weight_dict, f)
        
        f.close()

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

    def compute_covinv(self, residuals):
        
        """
        This functions computes the 4x4 covariance matrix of the residuals visibilities, 
        and it approximtes it inverse

        Args:
            residuals (np.array) : Array containing the residuals.
                              Shape is n_dir, n_tim, n_fre, n_ant, a_ant, n_cor, n_cor

        Returns:
            covinv (np.array) : Shape is ncor*n_cor x ncor*n_cor (4x4)
            Array containing the diagonal elements of the inverse covariance matrix
        """

        N = self.n_tim*self.n_fre*self.n_ant*self.n_ant

        res_reshaped = np.reshape(residuals,(N, 4))

        w = np.reshape(self.weights, (N,1))

        cov = res_reshaped.T.conjugate().dot(w*res_reshaped)/N

        covinv = np.linalg.pinv(cov)

        return np.array(covinv, dtype=self.dtype)


    def cycompute_weights(self, r, covinv, w, v, t_int, f_int):
        
        """
        This computes the weights, given the latest residual visibilities and the v parameter.
        w[i] = (v+2)/(v + 2*residual[i]^2). Next v is update using the newly compute weights.
        """
        
        def  _brute_solve_v(f, low, high):
            """finds the value of v by brute for using Gauss newton method"""
            root = None  # Initialization
            x = np.linspace(low, high, 58) #constraint the root to be between 2 and 30
            y = f(x)
    
            for i in range(len(x)-1):
                if y[i]*y[i+1] < 0:
                    root = x[i] - (x[i+1] - x[i])/(y[i+1] - y[i])*y[i]
                    break  # Jump out of loop
                elif y[i] == 0:       
                    root = x[i]
                    break  # Jump out of loop

            if root is None:
                dist = np.abs(y)
                root = x[np.argmin(dist)]
                print "Root not found, chosen value is %g"%root
                return root
            else:
                print 'Found a root, x=%g' % root
            return root

        n_mod = r.shape[0]
        n_tim = r.shape[1]
        n_fre = r.shape[2]
        n_ant = r.shape[3]

        for i in xrange(n_mod):
            for t in xrange(n_tim):
                for f in xrange(n_fre):
                    for aa in xrange(n_ant):
                        for ab in xrange(n_ant):
                            rr = np.reshape(r[i,t,f,aa,ab,:,:],(4))
                            w[i,t,f,aa,ab,0] = (v+8)/(v+ 2*rr.conj().T.dot(covinv.dot(rr)))
        
        #---------normalising to mean 1--------------------------#
        wnew = np.real(np.reshape(w[:,:,:,:,:,:],(n_tim*n_fre*n_ant*n_ant)))
        wnew_no_zero = wnew[np.where(wnew!=0)]
        norm = np.average(wnew_no_zero)
        w = w/norm              

        #-----------computing the v parameter---------------------#
        
        winit = np.real(np.reshape(w[:,:,:,:,:,:],(n_tim*n_fre*n_ant*n_ant)))
        wn = winit[np.where(winit!=0)]
        m = len(wn)

        vfunc = lambda a: special.digamma(0.5*(a+8)) - np.log(0.5*(a+8)) - special.digamma(0.5*a) + np.log(0.5*a) + (1./m)*np.sum(np.log(wn) - wn) + 1

        v = _brute_solve_v(vfunc, 2, 30)
        
        return w, v 

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

        flag_count = cyfull.cycompute_jhwjinv(self.gains, g_inv, self.gflags, self.eps, self.flagbit) # Function can invert G.

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
