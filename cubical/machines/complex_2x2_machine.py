# CubiCal: a radio interferometric calibration suite
# (c) 2017 Rhodes University & Jonathan S. Kenyon
# http://github.com/ratt-ru/CubiCal
# This code is distributed under the terms of GPLv2, see LICENSE.md for details
from __future__ import print_function
from cubical.machines.interval_gain_machine import PerIntervalGains
import numpy as np
import numpy.ma as ma
from numpy.ma import masked_array
from cubical.flagging import FL
import cubical.kernels

from cubical.tools import logger
log = logger.getLogger("complex_2x2")


class Complex2x2Gains(PerIntervalGains):
    """
    This class implements the full complex 2x2 gain machine.
    """
    def __init__(self, label, data_arr, ndir, nmod, chunk_ts, chunk_fs, chunk_label, options):
        """
        Initialises a 2x2 complex gain machine.
        
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
        # note that this sets up self.kernel
        PerIntervalGains.__init__(self, label, data_arr, ndir, nmod,
                                  chunk_ts, chunk_fs, chunk_label, options)

        self._estimate_pzd = options["estimate-pzd"]

        # this will be set to PZD and exp(-i*PZD) once the PZD estimate is done
        self._pzd = 0
        self._exp_pzd = 1

    @classmethod
    def determine_diagonality(cls, options):
        """Returns true if the machine class, given the options, represents a diagonal gain"""
        updates = set(options['update-type'].split("-"))
        return options['type'] == 'complex-diag' or \
               ('full' not in updates and 'leakage' not in updates)

    @staticmethod
    def exportable_solutions():
        """ Returns a dictionary of exportable solutions for this machine type. """
        sols = PerIntervalGains.exportable_solutions()
        sols["pzd"] = (0.0, ("time", "freq"))
        return sols

    def importable_solutions(self, grid0):
        """ Returns a dictionary of importable solutions for this machine type. """
        sols = super(Complex2x2Gains, self).importable_solutions(grid0)
        if "pzd" in self.update_type:
            sols["pzd"] = dict(**self.interval_grid)
        return sols

    def export_solutions(self):
        """ Saves the solutions to a dict of {label: solutions,grids} items. """
        sols = super(Complex2x2Gains, self).export_solutions()
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

        super(Complex2x2Gains, self).import_solutions(soldict)


    def compute_js(self, obser_arr, model_arr):
        """
        This function computes the (J\ :sup:`H`\J)\ :sup:`-1` and J\ :sup:`H`\R terms of the GN/LM
        method.

        Args:
            obser_arr (np.ndarray):
                Shape (n_mod, n_tim, n_fre, n_ant, n_ant, n_cor, n_cor) array containing the
                observed visibilities.
            model_arr (np.ndrray):
                Shape (n_dir, n_mod, n_tim, n_fre, n_ant, n_ant, n_cor, n_cor) array containing the
                model visibilities.

        Returns:
            3-element tuple

                - J\ :sup:`H`\R (np.ndarray)
                - (J\ :sup:`H`\J)\ :sup:`-1` (np.ndarray)
                - Count of flags raised (int)
        """

        n_dir, n_tim, n_fre, n_ant, n_cor, n_cor = self.gains.shape

        jh = self.get_new_jh(model_arr)

        self.kernel_solve.compute_jh(model_arr, self.gains, jh, self.t_int, self.f_int)

        jhr = self.get_new_jhr()
        r = self.get_obs_or_res(obser_arr, model_arr)

        if self.offdiag_only:
            jh[...,(0,1),(0,1)] = 0
            if r is obser_arr:
                r = r.copy()
            r[...,(0,1),(0,1)] = 0
            
        if self.diag_only:
            jh[...,(0,1),(1,0)] = 0
            if r is obser_arr:
                r = r.copy()
            r[...,(0,1),(1,0)] = 0

        self.kernel_solve.compute_jhr(jh, r, jhr, self.t_int, self.f_int)

        jhj, jhjinv = self.get_new_jhj()

        self.kernel_solve.compute_jhj(jh, jhj, self.t_int, self.f_int)

        if "scalar" in self.update_type:
            jhr[..., (0, 1), (0, 1)] = jhr[..., (0, 1), (0, 1)].mean(axis=-1)[..., np.newaxis]
            jhj[..., (0, 1), (0, 1)] = jhj[..., (0, 1), (0, 1)].mean(axis=-1)[..., np.newaxis]

        flag_count = self.kernel_solve.compute_jhjinv(jhj, jhjinv, self.gflags, self.eps, FL.ILLCOND)

#         if flag_count:
#             import pdb; pdb.set_trace()

        return jhr, jhjinv, flag_count

    #@profile
    def implement_update(self, jhr, jhjinv):

        # jhjinv is 2x2 block-diagonal, with Hermitian blocks. TODO: what's the variance on the off-diagonals?
        # variance of gain is diagonal of jhjinv
        if self.posterior_gain_error is None:
            self.posterior_gain_error = np.zeros_like(jhjinv.real)
        diag = jhjinv[..., (0, 1), (0, 1)].real
        self.posterior_gain_error[...,(0,1),(0,1)] = np.sqrt(diag)
        self.posterior_gain_error[...,(1,0),(0,1)] = np.sqrt(diag.sum(axis=-1)/2)[...,np.newaxis]

        update = self.init_update(jhr)

        self.kernel_solve.compute_update(jhr, jhjinv, update)

        if self.dd_term and self.n_dir > 1:
            update += self.gains

        self.restrict_solution(update)
        # raise flag so updates of G^H and G^-1 are computed
        self._gh_update = self._ghinv_update = True

        #if self._offdiag_only:
        #    update[...,0,0] = 1
        #    update[:,:,:,:,1,1] = self._exp_pzd[np.newaxis,:,:,np.newaxis]

        if self.iters % 2 == 0 or self.n_dir > 1:
            self.gains += update
            self.gains *= 0.5
            self.restrict_solution(self.gains)
        else:
            np.copyto(self.gains, update)

    def precompute_attributes(self, data_arr, model_arr, flags_arr, noise):
        """
        """
        super(Complex2x2Gains, self).precompute_attributes(data_arr, model_arr, flags_arr, noise)

        if self.solvable and self._estimate_pzd and self._pzd is 0 and model_arr is not None:
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
            with np.errstate(divide='ignore',invalid='ignore'):
                pzd = np.angle(dm_sum / dabs_sum)
            pzd[dabs_sum == 0] = 0
            with np.printoptions(precision=4, suppress=True, linewidth=1000):
                print("{0}: PZD estimate {1} deg".format(self.chunk_label, pzd * 180 / np.pi), file=log(2))
            self._pzd = pzd
            self._exp_pzd = np.exp(-1j * pzd)

            self.gains[:, :, :, :, 0, 0] = 1
            self.gains[:, :, :, :, 1, 1] = self._exp_pzd[np.newaxis, :, :, np.newaxis]


    def restrict_solution(self, gains):
        """
        Restricts the solution by invoking the inherited restrict_soultion method and applying
        any machine specific restrictions.
        """

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

        super(Complex2x2Gains, self).restrict_solution(gains)
        
        # with np.printoptions(precision=4, suppress=True):
        #     if self._jones_label == 'D':
        #         for p in range(self.n_ant):
        #             for a in (0,1):
        #                 for b in (0,1):
        #                     log.error("D{}{} for antenna {}: {}".format(a+1, b+1, p, gains[0,0,:,p,a,b]))

    @property
    def dof_per_antenna(self):
        if "leakage" in self.update_type and "pzd" in self.update_type:
            return 4 + 1./self.n_ant
        elif "leakage" in self.update_type:
            return 4
        elif "pzd" in self.update_type:
            return 1./self.n_ant
        else:
            return super(Complex2x2Gains, self).dof_per_antenna
