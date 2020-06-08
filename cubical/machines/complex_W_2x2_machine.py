# CubiCal: a radio interferometric calibration suite
# (c) 2017 Rhodes University & Jonathan S. Kenyon
# http://github.com/ratt-ru/CubiCal
# This code is distributed under the terms of GPLv2, see LICENSE.md for details
from __future__ import print_function
from cubical.machines.interval_gain_machine import PerIntervalGains
import numpy as np
from scipy import special, stats
from cubical.flagging import FL
import cubical.kernels
import time

from cubical.tools import logger, ModColor
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

        PerIntervalGains.__init__(self, label, data_arr, ndir, nmod, chunk_ts, chunk_fs, chunk_label, options)

        if self.is_diagonal:
            self.kernel_robust = cubical.kernels.import_kernel("diag_robust")
        else:
            self.kernel_robust = cubical.kernels.import_kernel("full_W_complex")

        self.residuals = np.empty_like(data_arr)

        self.save_weights = options.get("robust-save-weights", False)
        
        self.label = label

        self.cov_type = options.get("robust-cov", "compute") # adding an option to compute residuals covariance or just assume 1 as in Robust-t paper

        self.npol = options.get("robust-npol", 2.) # testing if the number of polarizations really have huge effects

        self.v_int = options.get("robust-int", 5)

        self.cov_scale = options.get("robust-scale", 0) # scale down the covariance by this factor

        self.cov_thresh = options.get("robust-cov-thresh",  1)

        self.sigma_thresh = options.get("robust-sigma-thresh", 3)

        self.robust_flag_weights = options.get("robust-flag-weights", True)

        self.fixed_v = False

        self.not_all_flagged = True

        self._flag = True

        self.any_new = 0

        self.is_robust = True # to identify this machine as the robust solver

        self._estimate_pzd = options["estimate-pzd"]

        # this will be set to PZD and exp(-i*PZD) once the PZD estimate is done
        self._pzd = 0
        self._exp_pzd = 1


    @classmethod
    def determine_diagonality(cls, options):
        """Returns true if the machine class, given the options, represents a diagonal gain"""
        updates = set(options['update-type'].split("-"))
        return options['type'] == 'robust-diag' or \
               ('full' not in updates and 'leakage' not in updates)

    @staticmethod
    def exportable_solutions():
        """ Returns a dictionary of exportable solutions for this machine type. """
        sols = PerIntervalGains.exportable_solutions()
        sols["pzd"] = (0.0, ("time", "freq"))
        return sols

    def importable_solutions(self, grid0):
        """ Returns a dictionary of importable solutions for this machine type. """
        sols = super(ComplexW2x2Gains, self).importable_solutions(grid0)
        if "pzd" in self.update_type:
            sols["pzd"] = dict(**self.interval_grid)
        return sols

    def export_solutions(self):
        """ Saves the solutions to a dict of {label: solutions,grids} items. """
        sols = super(ComplexW2x2Gains, self).export_solutions()
        if "pzd" in self.update_type and self._pzd is not None:
            sols["pzd"] = (masked_array(self._pzd), self.interval_grid)
        return sols

    def import_solutions(self, soldict):
        """
        Loads solutions from a dict.

        Args:
            soldict (dict):
                Contains gains solutions which must be loaded.
        """
        if "pzd" in self.update_type and "pzd" in soldict:
            self._pzd = soldict["pzd"]
            self._exp_pzd = np.exp(-1j * self._pzd)

        super(ComplexW2x2Gains, self).import_solutions(soldict)

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
        # w = self.weights
        
        n_dir, n_tim, n_fre, n_ant, n_cor, n_cor = self.gains.shape

        # compute residuals and weights
        if self.iters == 1:
            self.residuals = self.compute_residual(obser_arr, model_arr, self.residuals)

            if self.not_all_flagged:
                self.update_weights()


        jh = self.get_new_jh(model_arr)

        if self.offdiag_only:
            jh[...,(0,1),(0,1)] = 0
            self.residuals[...,(0,1),(0,1)] = 0
            
        if self.diag_only:
            jh[...,(0,1),(1,0)] = 0
            self.residuals[...,(0,1),(1,0)] = 0

        self.kernel_robust.compute_jh(model_arr, self.gains, jh, self.t_int, self.f_int)

        jhwr = self.get_new_jhr()

        self.kernel_robust.compute_jhwr(jh, self.residuals, self.weights, jhwr, self.t_int, self.f_int) #TODO 

        jhwj, jhwjinv = self.get_new_jhj()

        self.kernel_robust.compute_jhwj(jh, self.weights, jhwj, self.t_int, self.f_int)

        flag_count = self.kernel_robust.compute_jhjinv(jhwj, jhwjinv, self.gflags, self.eps, FL.ILLCOND)
        
        return jhwr, jhwjinv, flag_count

    
    #@profile
    def implement_update(self, jhr, jhjinv):

        # jhjinv is 2x2 block-diagonal, with Hermitian blocks. TODO: what's the variance on the off-diagonals?
        # variance of gain is diagonal of jhjinv
        # not sure about how the  weights affects the posterior variance. here we actually pass jhwj, not jhj
        if self.posterior_gain_error is None:
            self.posterior_gain_error = np.zeros_like(jhjinv.real)
        
        #----normalising to reduce the effects of the weights on jhj---#
        #----not sure if this is the best way to do this---------------#

        self.Nvis = np.sum(self.new_flags==False)
        try:
            norm = np.min(self.weights[np.where(self.weights>0)]).real #np.real((1/2.)*np.sum(self.weights)) #/self.Nvis
        except:
            norm = 0
        diag = norm*jhjinv[..., (0, 1), (0, 1)].real #/np.max(self.weights)
        self.posterior_gain_error[...,(0,1),(0,1)] = np.sqrt(diag)
        self.posterior_gain_error[...,(1,0),(0,1)] = np.sqrt(diag.sum(axis=-1)/2)[...,np.newaxis]

        update = self.init_update(jhr)
        self.kernel_robust.compute_update(jhr, jhjinv, update)


        # if self.dd_term and self.n_dir > 1: computing residuals for both DD and DID calibration
        update += self.gains
        self.restrict_solution(update)

        if self.iters % 2 == 0 or self.n_dir > 1:
            self.gains += update
            self.gains *= 0.5
            self.restrict_solution(self.gains)
        else:
            np.copyto(self.gains, update)


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

        if self.not_all_flagged:
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

            Nvis = self.Nvis/2. #only half of the visibilties are used for covariance computation

            ompstd = np.zeros((4,4), dtype=self.dtype)

            self.kernel_robust.compute_cov(self.residuals, ompstd, self.weights)

            # removing the offdiagonal correlations
            std = np.diagonal(ompstd/Nvis) + self.eps**2 # To avoid division by zero

            if np.any(std>self.cov_thresh):
                self.fixed_v = True
                
                if self.flaground:
                    print(ModColor.Str("rb-2x2 {} : flag round {}: Warning Covariance too high probably because of RFI will fixed v to 2 and cov to 1".format(self.label, self._count+1), "red"), file=log(2))
                else:
                    print(ModColor.Str("rb-2x2 {} : {} iters: Warning Covariance too high probably because of RFI will fixed v to 2 and cov to 1".format(self.label, self.iters), "red"), file=log(2))

            else:
                #---scaling the variance in this case improves the robust solver performance----------#
                self.fixed_v = False
                if self.cov_scale and not self.flaground:
                    std /= self.cov_scale 

            if self.iters % 5 == 0 or self.iters == 1:
                if self.flaground:
                    print("rb-2x2 {} : flag round {}: covariance diagonal : [{}]".format(self.label, self._count+1, ", ".join('{:.2g}'.format(x) for x in std)), file=log(2))
                    # print("rb-2x2 {} : flag round {}: covariance diagonal : [{}]".format(self.label, self._count+1, ", ".join('{:.2g}'.format(x) for x in std2)), file=log(2))
                else:
                    print("rb-2x2 {} : {} iters: covariance diagonal : [{}]".format(self.label, self.iters, ", ".join('{:.2g}'.format(x) for x in std)), file=log(2))
                    # print("rb-2x2 {} : {} iters: covariance diagonal : [{}]".format(self.label, self.iters, ", ".join('{:.2g}'.format(x) for x in std2)), file=log(2))


            # can we disable the flagging the solver to avoid flagging unmodelled sources
            # UMS: my thought here is that if data is unmodelled sources rather than RFI xx and yy covariance should be very close
            # so the solver should disable the flagging in this case
            xx_close_to_yy = 0.8 <= np.abs(std[0])/np.abs(std[0]) <= 1.2
            cov_low = np.average([std[0], std[3]]).real < 2e-2

            if xx_close_to_yy and cov_low and self.flaground and self._count==0:
                self.robust_flag_disable = True
                print(ModColor.Str("rb-2x2 {} : flag round {}: Warning: the covariance is low and the xx and yy variances are very close. Flagging will be disable".format(self.label, self._count+1), "red"), file=log(2))


            covinv = np.eye(4, dtype=self.dtype)

            if self.fixed_v:
                covinv *= 1/self.cov_thresh
            else:
                if self.cov_type == "hybrid":
                    if np.max(np.abs(std)) < 1:
                        covinv[np.diag_indices(4)]= 1/std
                
                elif self.cov_type == "compute":
                    covinv[np.diag_indices(4)]= 1/std
                else:
                    raise RuntimeError("unknown robust-cov setting")
        
        if self.npol == 2:
            covinv[(1,2), (1,2)] = 0


        self.covinv = covinv

        # print("rb-2x2 {} : {} iters: hybrid check covariance diagonal : [{}]".format(self.label, self.iters, ", ".join('{:.2g}'.format(x) for x in np.diagonal(covinv))), file=log(2))


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
                wn : the weights flatenncc-a-b in to a 1D array
            Returns:
                root (float) : The root of f or minimum point    
            """
            # import pdb; pdb.set_trace()

            m = len(wn)

            
            vfunc = lambda a: special.digamma(a+self.npol) - np.log(a+self.npol) - \
                        special.digamma(a) + np.log(a) + (1./m)*np.sum(np.log(wn) - wn) + 1
            

            vvals = np.arange(2, 101, 1, dtype=float) #search for v in the range (2, 100)
            fvals = vfunc(vvals)
            root = vvals[np.argmin(np.abs(fvals))]

            if self.flaground:
                print("rb-2x2 {} : flag round {}: v-parameter is  {:.3}".format(self.label, self.iters, root), file=log(2))
                print("rb-2x2 {} : flag round {} : weights stats are min {:.4}, max: {:.4}, mean: {:.4}, std: {:.4}, median: {:.4}, mad: {:.4}".format(self.label, self._count+1, np.min(wn), 
                                                    np.max(wn), np.mean(wn), np.std(wn), np.median(wn), stats.median_absolute_deviation(wn)), file=log(2)) 
            else:
                
                if self.iters % 5 == 0 or self.iters == 1:
                    print("rb-2x2 {} : {} iters: v-parameter is  {:.3}".format(self.label, self.iters, root), file=log(2))
                    #nless, nmore = len(wn[wn<1])/len(wn), len(wn[wn>1])/len(wn)
                    print("rb-2x2 {} : {} iters: weights stats are min {:.4}, max: {:.4}, mean: {:.4}, std: {:.4}, median: {:.4}, mad: {:.4}".format(self.label, self.iters, np.min(wn), 
                                                    np.max(wn), np.mean(wn), np.std(wn), np.median(wn), stats.median_absolute_deviation(wn)), file=log(2)) #, np.sum(wn), m, nless, nmore, 
            
            if self.fixed_v:
                return 2
            else:    
                return root

        if self.iters % self.v_int == 0 or self.iters == 1:
            self.compute_covinv()

        if self.fixed_v:
            self.v = 2

        self.kernel_robust.compute_weights(self.residuals, self.covinv, self.weights, self.v, self.npol)

        # re-set weights for visibillities flagged from start to 0 and normalise the weights by their sum 
        self.weights[:,self.new_flags!=0,:] = 0
        
        aa, ab = np.tril_indices(self.n_ant, -1)
        w_real = np.real(self.weights[:,:,:,aa,ab,0].flatten())
        w_nzero = w_real[np.where(w_real!=0)[0]]
        norm = np.average(w_nzero)

        self.weights/= norm
        w_nzero/=norm
        
        # wstd = np.std(w_nzero) #stats.median_absolute_deviation(w_nzero)
        # wmean = 1 #np.median(w_nzero)
        
        #-----------computing the v parameter---------------------#
        # This computation is only done after a certain number of iterations. Default is 5
        if self.iters % self.v_int == 0 or self.iters == 1:
            self.v = _brute_solve_v(w_nzero) # remove zero weights

        self.not_all_flagged = self.flag_weights()
        
        return

    def flag_weights(self):
        """Trying flagging visiblities with very very low weights and see if this improves the solution
        like mad max flagger

        wstd: the weights standard deviation
        """

        if self.flaground and not self.robust_flag_disable:

            if self._final_flaground:
                _pre_or_post = "after-solving"   
            else:
                _pre_or_post = "before solving"

            _nvis = self.new_flags.size
            nflag0 = np.sum(self.new_flags!=0)
            wfrac0 = nflag0/_nvis

            # This correction factor is two ensure that we only flag when v is low.
            # The 1.26 factor is just makes it work somehow 
            _v_corr = 1.26/self.v #2*self.npol/self.v if self.v < 5 else self.npol/self.v

            wlow =  _v_corr*(self.v + self.npol)/(self.v + self.sigma_thresh)

            # print("rb-2x2 {} : {} iters: wlow is  {:.3} while 1/v is {:.3}".format(self.label, self.iters, wlow, 1/self.v), file=log(2))

            self.weight_flags = np.where((self.weights< wlow) & (self.weights!=0)) # wlow

            # import pdb; pdb.set_trace()

            if len(self.weight_flags[0])>0:

                self.weights[self.weight_flags] = 0
                self.new_flags[self.weight_flags[1:-1]] |= FL.MAD
                self.residuals[self.weight_flags[:-1]] = 0

                nflag = np.sum(self.new_flags!=0)

                any_new = nflag-nflag0

                if any_new:
                    wfrac = any_new/_nvis
                    
                    print(ModColor.Str("rb-2x2 {} : {} flag round {} : number of weight flags {} ({:.4%}), prior flags {} ({:.4%})".format(self.label, _pre_or_post, self._count+1, any_new, wfrac, nflag0, wfrac0), "blue"), file=log(2))
                
                self.any_new = True

                self._count += 1 
                
                return False if nflag ==_nvis else True

            else:
                if self._count==0:
                    self.any_new = False 
                return True
        
        else:
            self.any_new = False
            self.weight_flags = None 
            
            return True


    def update_weight_flags(self, flags_arr):

        self.new_flags = flags_arr
        self.weights[:,self.new_flags!=0,:] = 0
        
        self.Nvis = np.sum(self.new_flags==False)

        self.not_all_flagged = False if self.Nvis==0 else True

    def robust_flag(self, flags_arr, model_arr, obser_arr, final=False):
        """run an iteration and use the weights to flag high level RFIs"""

        self.flaground = True

        if final is False:
            self.v = 5  # Don't start with a low degree of freedom to prevent overflagging
            self.iters = 1

        self._final_flaground = True if final else False

        if final and self.not_all_flagged:
            self.flag_weights()
        else:
            self.compute_update(model_arr, obser_arr)
            np.copyto(self.gains, self.old_gains)

        if self.any_new:
            flags_arr[self.weight_flags[1:-1]] |= FL.MAD
            self.update_equation_counts(flags_arr != 0)

            new_flags = flags_arr & ~(FL.MISSING | FL.PRIOR) != 0
                
            model_arr[:, :, new_flags!=0, :, :] = 0
            obser_arr[   :, new_flags!=0, :, :] = 0

        if final is False:
            self.weights[:, self.new_flags==0, :] = 1
            self.iters = 0
            self.v = 2
            self._gh_update = self._ghinv_update = True

        self.flaground = False

    def restrict_solution(self, gains):
        
        
        if "pzd" in self.update_type:
            # re-estimate pzd
            mask = self.gflags!=0
            with np.errstate(divide='ignore', invalid='ignore'):
                pzd = masked_array(gains[:, :, :, :, 0, 0] / gains[:, :, :, :, 1, 1], mask)
            pzd = np.angle(pzd.sum(axis=(0,3)))
            with np.printoptions(precision=4, suppress=True, linewidth=1000):
                print("{0}: PZD estimate changes by {1} deg".format(self.chunk_label, (pzd-self._pzd)* 180 / np.pi), file=log(2))
            # import ipdb; ipdb.set_trace()
            self._pzd = pzd
            self._exp_pzd = np.exp(-1j * pzd)

            gains[:, :, :, :, 0, 0] = 1
            gains[:, :, :, :, 1, 1] = self._exp_pzd[np.newaxis, :, :, np.newaxis]
            
        if "leakage" in self.update_type:
            if "pzd" not in self.update_type:
                gains[:, :, :, :, 0, 0] = 1
                gains[:, :, :, :, 1, 1] = 1
                
            if "rel" in self.update_type and self.ref_ant is not None:
                offset =  gains[:, :, :, self.ref_ant, 0, 1].copy()
                gains[..., 0, 1] -= offset[..., np.newaxis]
                gains[..., 1, 0] += np.conj(offset)[..., np.newaxis]
                with np.printoptions(precision=4, suppress=True, linewidth=1000):
                    print("{0}: subtracting relative leakage offset {1}".format(self.chunk_label, offset), file=log(2))


        if self.ref_ant is not None:
            phase = np.angle(self.gains[...,self.ref_ant,0,0])
            gains[:,:,:,:,(0,1),(0,1)] *= np.exp(-1j*phase)[:,:,:,np.newaxis,np.newaxis]

        super(ComplexW2x2Gains, self).restrict_solution(gains)


    def precompute_attributes(self, data_arr, model_arr, flags_arr, noise):
        """
        Set the initial weights to 1 and set the weights of the flags data points to 0

        Args:
            model_arr (np.ndarray):
                Shape (n_dir, n_mod, n_tim, n_fre, n_ant, n_ant, n_cor, n_cor) array containing 
                model visibilities.
            flags_arr (np.ndarray):
                Shape (n_tim, n_fre, n_ant, n_ant) array containing  flags
        """
        
        super(ComplexW2x2Gains, self).precompute_attributes(data_arr, model_arr, flags_arr, noise)

        if self._estimate_pzd and self._pzd is 0:
            marr = model_arr[..., (0, 1), (1, 0)][:, 0].sum(0)
            darr = data_arr[..., (0, 1), (1, 0)][0]
            mask = (flags_arr[..., np.newaxis] != 0) | (marr == 0)
            with np.errstate(divide='ignore', invalid='ignore'):
                dm = darr * (np.conj(marr) / abs(marr))
            dabs = np.abs(darr)
            dm[mask] = 0
            dabs[mask] = 0
            # collapse time/freq axis into intervals and sum antenna axes
            dm_sum = self.interval_sum(dm).sum(axis=(2, 3))
            dabs_sum = self.interval_sum(dabs).sum(axis=(2, 3))
            # sum off-diagonal terms
            dm_sum = dm_sum[..., 0] + np.conj(dm_sum[..., 1])
            dabs_sum = dabs_sum[..., 0] + np.conj(dabs_sum[..., 1])
            pzd = np.angle(dm_sum / dabs_sum)
            pzd[dabs_sum == 0] = 0
            with np.printoptions(precision=4, suppress=True, linewidth=1000):
                print("{0}: PZD estimate {1} deg".format(self.chunk_label, pzd * 180 / np.pi), file=log(2))
            self._pzd = pzd
            self._exp_pzd = np.exp(-1j * pzd)

            self.gains[:, :, :, :, 0, 0] = 1
            self.gains[:, :, :, :, 1, 1] = self._exp_pzd[np.newaxis, :, :, np.newaxis]

        self.weights_shape = [self.n_mod, self.n_tim, self.n_fre, self.n_ant, self.n_ant, 1]
        
        self.weights = np.ones(self.weights_shape, dtype=self.dtype)

        self.v = 2 # t-distribution number of degrees of freedom

        self.covinv = np.eye(4, dtype=self.dtype)

        self.weight_flags = None

        self.update_weight_flags(flags_arr)

        self.flaground = False

        self._final_flaground = False

        self.robust_flag_disable = False

        self._count = 0

    @property
    def dof_per_antenna(self):
        if "leakage" in self.update_type and "pzd" in self.update_type:
            return 4 + 1./self.n_ant
        elif "leakage" in self.update_type:
            return 4
        elif "pzd" in self.update_type:
            return 1./self.n_ant
        else:
            return super(ComplexW2x2Gains, self).dof_per_antenna
