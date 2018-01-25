# CubiCal: a radio interferometric calibration suite
# (c) 2017 Rhodes University & Jonathan S. Kenyon
# http://github.com/ratt-ru/CubiCal
# This code is distributed under the terms of GPLv2, see LICENSE.md for details
from cubical.machines.interval_gain_machine import PerIntervalGains
import numpy as np
import cubical.kernels.cyfull_W_complex as cyfull
from scipy import special
from scipy.optimize import fsolve


class ComplexW2x2Gains(PerIntervalGains):
    """
    This class implements the weighted full complex 2x2 gain machine based on the Complex T-distribution
    """
    
    def __init__(self, label, model_arr, ndir, nmod, chunk_ts, chunk_fs, options):
        
        PerIntervalGains.__init__(self, label, model_arr, ndir, nmod, chunk_ts, chunk_fs, options)
        
        self.gains     = np.empty(self.gain_shape, dtype=self.dtype)
        
        self.gains[:]  = np.eye(self.n_cor)
        
        self.old_gains = self.gains.copy()
        
        self.weights_shape = [self.n_mod, self.n_tim, self.n_fre, self.n_ant, self.n_ant, 1]
        
        self.weights = np.ones(self.weights_shape, dtype=self.dtype)
        self.weights[:,:,:,(range(self.n_ant),range(self.n_ant)),0] = 0 #setting the initial weights for the autocorrelations 0
        
        self.v = 2.
        self.weight_dict = {}
        self.weight_dict["weights"] = {}
        self.weight_dict["vvals"] = {}
        self.save_weights = options["save_weights"] if "save_weights" in options.keys() else False
        
        self.label = label

        self.cov_type = options["cov_type"] if "cov_type" in options.keys() else 1 #adding an option to compute residuals covariance or just assume 1 as in Robust-t paper

        self.npol = options["npol"] if "npol" in options.keys() else 4 #testing if the number of polarizations really have huge effects
        
    def compute_js(self, obser_arr, model_arr):
        """
        This function computes the (J^H)WR term of the weighted GN/LM method for the
        full-polarisation, phase-only case.

        Args:
            obser_arr (np.array): Array containing the observed visibilities.
            model_arr (np.array): Array containing the model visibilities.
            gains (np.array): Array containing the current gain estimates.

        Returns:
            Returns:
            jhwr (np.array): Array containing the result of computing (J^H)WR.
            jhwjinv (np.array): Array containing the result of computing (J^HW.J)^-1
            flag_count:     Number of flagged (ill-conditioned) elements
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

    def compute_update(self, model_arr, obser_arr):
        """
        This function computes the update step of the weighted GN/LM method. This is
        equivalent to the complete (((J^H)WJ)^-1)(J^H)WR.

        Args:
            obser_arr (np.array): Array containing the observed visibilities.
            model_arr (np.array): Array containing the model visibilities.
            gains (np.array): Array containing the current gain estimates.
            jhwjinv (np.array): Array containing (J^H)WJ)^-1. (Invariant)

        Returns:
            update (np.array): Array containing the result of computing
                (((J^H)WJ)^-1)(J^H)WR
        """

        
        jhwr, jhwjinv, flag_count = self.compute_js(obser_arr, model_arr)

        update = np.empty_like(jhwr)

        cyfull.cycompute_update(jhwr, jhwjinv, update)

        if model_arr.shape[0]>1 or self.n_dir>1:
			update = self.gains + update

        if self.iters % 2 == 0 or self.n_dir>1 :
            self.gains = 0.5*(self.gains + update)
        else:
            self.gains = update

        self.restrict_solution()
        
        #Computing the weights
        resid_arr = np.empty_like(obser_arr)
        
        residuals = self.compute_residual(obser_arr, model_arr, resid_arr)

        covinv = self.compute_covinv(residuals)
        
        self.weights, self.v = self.update_weights(residuals, covinv, self.weights, self.v)

        if self.save_weights:
            self.weight_dict["weights"][self.iters] = self.weights
            
            self.weight_dict["vvals"][self.iters] = self.v
            
            np.savez("./"+ self.label + "_weights_dict.npz", **self.weight_dict)
        

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
        and it approximtes it inverse. I self.cov_type is set to 1, the covariance maxtrix is 
        assumed to be the Identity matrix as in the Robust-t paper.

        Args:
            residuals (np.array) : Array containing the residuals.
                Shape is n_dir, n_tim, n_fre, n_ant, a_ant, n_cor, n_cor

        Returns:
            covinv (np.array) : Shape is ncor*n_cor x ncor*n_cor (4x4)
            Array containing the inverse covariance matrix

        """

        if self.cov_type == 1:
            
            covinv = np.eye(4, dtype=self.dtype)
        
        else:
            N = self.n_tim*self.n_fre*self.n_ant*self.n_ant

            res_reshaped = np.reshape(residuals[:,:,:,:,:,(0,1),(0,1)],(2, N))

            w = np.reshape(self.weights, (N))

            std = np.cov(res_reshaped, aweights=w) #[0,0] #just return the first element as diag
            stdinv = np.linalg.pinv(std)

            print "std and inverse", std , " ", stdinv
            
            covinv = np.eye(4, dtype=self.dtype) #1/std)*
            covinv[0,0] = stdinv[0,0]
            covinv[0,3] = stdinv[0,1]
            covinv[3,0] = stdinv[1,0] 
            covinv[3,3] = stdinv[1,1]

        if self.npol == 2:
            
            covinv[(1,2),(1,2)] = 0

        return covinv


    def update_weights(self, r, covinv, w, v):
        
        """
            This computes the weights, given the latest residual visibilities and the v parameter.
            w[i] = (v+2*npol)/(v + 2*r[i].T.cov.r[i]. Next v is update using the newly compute weights.
        
            Args:
                r (np.array): Array of the residual visibilities.
                covinc (np.array) : Array containing the inverse of 
               covariance of residual visibilities
                w (np.array): Array containing the weights
                v (float) : v -- number of degrees of freedom

            Returns:
                w (np.array) : new weights
                v (float) : new value for v
        """

        def  _brute_solve_v(f, low, high):
            """Finds a root for the function f constraint between low and high
            Args:
                f (callable) : function
                low (float): lower bound, 2.
                high (float): upper bound, 30.

            Returns:
                root (float) : The root of f or minimum point    
            """

            root = fsolve(f, 2.)[0]
            if root < low:
                root = low
            elif root > high:
                root= high
            else:
                root = root

            print "v from fsolve", root
            
            return root

        cyfull.cycompute_weights(r,covinv,w,v,self.npol)
        w[:,:,:,(range(self.n_ant),range(self.n_ant)),0] = 0  #setting the weights for the autocorrelations 0

        #---------normalising the weights to mean 1 --------------------------#
        w_real = np.real(w.flatten())
        w_nzero = w_real[np.where(w_real!=0)[0]]  #removing the autocorrelatios zero weights
        norm = np.average(w_nzero)
        w = w/norm           

        #-----------computing the v parameter---------------------#
        wn = w_nzero/norm
        m = len(wn)

        print (wn[np.where(wn<0)[0]])

        vfunc = lambda a: special.digamma(0.5*(a+2*self.npol)) - np.log(0.5*(a+2*self.npol)) - special.digamma(0.5*a) + np.log(0.5*a) + (1./m)*np.sum(np.log(wn) - wn) + 1

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

    def apply_gains(self, model_arr):
        """
        This method should be able to apply the gains to an array at full time-frequency
        resolution. Should return the input array at full resolution after the application of the 
        gains.
        """

        gh = self.gains.transpose(0,1,2,3,5,4).conj()

        cyfull.cyapply_gains(model_arr, self.gains, gh, self.t_int, self.f_int)

        return model_arr

    def restrict_solution(self):
        
        PerIntervalGains.restrict_solution(self)

        if self.ref_ant is not None:
            phase = np.angle(self.gains[...,self.ref_ant,(0,1),(0,1)])
            self.gains *= np.exp(-1j*phase)[:,:,:,np.newaxis,:,np.newaxis]