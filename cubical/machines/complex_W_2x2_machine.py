# CubiCal: a radio interferometric calibration suite
# (c) 2017 Rhodes University & Jonathan S. Kenyon
# http://github.com/ratt-ru/CubiCal
# This code is distributed under the terms of GPLv2, see LICENSE.md for details
from cubical.machines.interval_gain_machine import PerIntervalGains
import numpy as np
from scipy import special
from cubical.flagging import FL
import cubical.kernels
import time

from cubical.tools import logger
log = logger.getLogger("robust_2x2")  #TODO check this "complex_2x2"

class ComplexW2x2Gains(PerIntervalGains):
    """
    This class implements the weighted full complex 2x2 gain machine based on the Complex T-distribution
    """
    
    def __init__(self, label, data_arr, ndir, nmod,
                 chunk_ts, chunk_fs, chunk_label, options):

        """
        Initialises a weighted complex 2x2 gain machine.
        
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
        
        

        # clumsy but necessary: can't import at top level (OMP must not be touched before worker processes
        # are forked off), so we import it only in here

        
        PerIntervalGains.__init__(self, label, data_arr, ndir, nmod, chunk_ts, chunk_fs, chunk_label, 
                                    options, self.get_kernel(options))

        self.residuals = np.empty_like(data_arr)

        self.save_weights = options.get("robust-save-weights", False)
        
        self.label = label

        self.cov_type = options.get("robust-cov", "compute") #adding an option to compute residuals covariance or just assume 1 as in Robust-t paper

        self.npol = options.get("robust-npol", 2.) #testing if the number of polarizations really have huge effects

        self.v_int = options.get("robust-int", 1)

        self.cov_scale = options.get("robust-cov-scale", True) # scale the covariance by n_corr*2

       
    @staticmethod
    def get_kernel(options):
        """Returns kernel approriate to Jones options"""
        return cubical.kernels.import_kernel('cyfull_W_complex') #TODO : check this import 
        
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

        jh = self.get_new_jh(model_arr)

        self.cykernel.cycompute_jh(model_arr, self.gains, jh, self.t_int, self.f_int)

        jhwr = self.get_new_jhr()

        if self.iters == 1:
            self.residuals = self.compute_residual(obser_arr, model_arr, self.residuals)

        self.cykernel.cycompute_jhwr(jh, self.residuals, w, jhwr, self.t_int, self.f_int) #TODO 

        jhwj, jhwjinv = self.get_new_jhj()

        self.cykernel.cycompute_jhwj(jh, w, jhwj, self.t_int, self.f_int)

        flag_count = self.cykernel.cycompute_jhjinv(jhwj, jhwjinv, self.gflags, self.eps, FL.ILLCOND)

        return jhwr, jhwjinv, flag_count

    
    #@profile
    def implement_update(self, jhr, jhjinv):

        # jhjinv is 2x2 block-diagonal, with Hermitian blocks. TODO: what's the variance on the off-diagonals?
        # variance of gain is diagonal of jhjinv
        # not sure about how the  weights affects the posterior variance. here we actually pass jhwj, not jhj

        if self.posterior_gain_error is None:
            self.posterior_gain_error = np.zeros_like(jhjinv.real)
        diag = jhjinv[..., (0, 1), (0, 1)].real
        self.posterior_gain_error[...,(0,1),(0,1)] = np.sqrt(diag)
        self.posterior_gain_error[...,(1,0),(0,1)] = np.sqrt(diag.sum(axis=-1)/2)[...,np.newaxis]

        update = self.init_update(jhr)
        self.cykernel.cycompute_update(jhr, jhjinv, update)

        # if self.dd_term and self.n_dir > 1: computing residuals for both DD and DID calibration
        update += self.gains

        if self.iters % 2 == 0 or self.n_dir > 1:
            self.gains += update
            self.gains *= 0.5
        else:
            np.copyto(self.gains, update)

        self.restrict_solution()

    def compute_update(self, model_arr, obser_arr):
        """
        This method is expected to compute the parameter update. 
        
        The standard implementation simply calls compute_js() and implement_update(), here we are
        overriding it because we need to update weights as well. 

        Args:
            model_arr (np.ndarray): 
                Shape (n_dir, n_mod, n_tim, n_fre, n_ant, n_ant, n_cor, n_cor) array containing 
                model visibilities. 
            
            obser_arr (np.ndarray): 
                Shape (n_mod, n_tim, n_fre, n_ant, n_ant, n_cor, n_cor) array containing observed 
                visibilities. 
        """
        
        jhwr, jhwjinv, flag_count = self.compute_js(obser_arr, model_arr)

        self.implement_update(jhwr, jhwjinv)

        # Computing the weights
        
        self.residuals = self.compute_residual(obser_arr, model_arr, self.residuals)

        self.update_weights()

        return flag_count

    def compute_covinv(self):
        
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

        if self.cov_type == "identity":

            covinv = np.eye(4, dtype=self.dtype)
          
        else:

            unflagged = self.new_flags==False 

            Nvis = np.sum(unflagged)/2. #only half of the visibilties are used for covariance computation

            ompstd = np.zeros((4,4), dtype=self.dtype)

            self.cykernel.cycompute_cov(self.residuals, ompstd, self.weights)

            #---if the covariance and variance are close the residuals are dominated by sources---#
            #---scaling the variance in this case improves the robust solver performance----------#
            
            if self.cov_scale:
                norm = 2*self.npol*Nvis
            else:
                norm =Nvis

            if self.iters % 5 == 0 or self.iters == 1:
                print("{} : {} iters: covariance is  {}".format(self.label, self.iters, ompstd/Nvis), file=log(2))

            # removing the offdiagonal correlations

            std = np.diagonal(ompstd/norm) + self.eps**2 # To avoid division by zero

            covinv = np.eye(4, dtype=self.dtype)

            if self.cov_type == "hybrid":
                if np.max(np.abs(std)) < 1:
                    covinv[np.diag_indices(4)]= 1/std
                    
            elif self.cov_type == "compute":
                covinv[np.diag_indices(4)]= 1/std
               
            else:
                raise RuntimeError("unknown robust-cov setting")
        
        if self.npol == 2:
            covinv[(1,2), (1,2)] = 0
            

        return covinv
    

    def update_weights(self):
        
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

        def  _brute_solve_v(wn):
            """Finds a root for the function f constraint between low (2) and high (50)
            Args:
                wn : the weights flaten in to a 1D array
            Returns:
                root (float) : The root of f or minimum point    
            """

            m = len(wn)

            
            vfunc = lambda a: special.digamma(a+self.npol) - np.log(a+self.npol) - \
                        special.digamma(a) + np.log(a) + (1./m)*np.sum(np.log(wn) - wn) + 1
            

            vvals = np.arange(2, 101, 1, dtype=float) #search for v in the range (2, 100)
            fvals = vfunc(vvals)
            root = vvals[np.argmin(np.abs(fvals))]

            if self.iters % 5 == 0 or self.iters == 1:
                print>> log(2), "{} : {} iters: v-parameter is  {}".format(self.label, self.iters, root)
            
            return root

        covinv = self.compute_covinv()

        w , v  = self.weights, self.v

        # import pdb; pdb.set_trace()

        self.cykernel.cycompute_weights(self.residuals, covinv, w, v, self.npol)

        # re-set weights for visibillities flagged from start to 0
        w[:,self.new_flags,:] = 0

        #---------normalising the weights to mean 1 using only half the weights--------------------------#
        aa, ab = np.tril_indices(self.n_ant, -1)
        w_real = np.real(w[:,:,:,aa,ab,0].flatten())
        w_nzero = w_real[np.where(w_real!=0)[0]]  #removing zero weights for the v computation
        
        # norm = np.average(w_nzero) 
  
        self.weights = w #/norm #removed normalisation
        
        #-----------computing the v parameter---------------------#
        # This computation is only done after a certain number of iterations. Default is 5
        if self.iters % self.v_int == 0 or self.iters == 1:
            wn = w_nzero #/norm 
            self.v = _brute_solve_v(wn)
        
        return 

    def restrict_solution(self):
        
        PerIntervalGains.restrict_solution(self)

        if self.ref_ant is not None:
            phase = np.angle(self.gains[...,self.ref_ant,(0,1),(0,1)])
            self.gains *= np.exp(-1j*phase)[:,:,:,np.newaxis,:,np.newaxis]



    def precompute_attributes(self, model_arr, flags_arr, noise):
        """
        Set the initial weights to 1 and set the weights of the flags data points to 0

        Args:
            model_arr (np.ndarray):
                Shape (n_dir, n_mod, n_tim, n_fre, n_ant, n_ant, n_cor, n_cor) array containing 
                model visibilities.
            flags_arr (np.ndarray):
                Shape (n_tim, n_fre, n_ant, n_ant) array containing  flags
        """
        PerIntervalGains.precompute_attributes(self, model_arr, flags_arr, noise)

        self.weights_shape = [self.n_mod, self.n_tim, self.n_fre, self.n_ant, self.n_ant, 1]
        
        self.weights = np.ones(self.weights_shape, dtype=self.dtype)
        self.weights[:,flags_arr!=0] = 0
        self.new_flags = flags_arr!=0

        self.v = 2.   # t-distribution number of degrees of freedom
