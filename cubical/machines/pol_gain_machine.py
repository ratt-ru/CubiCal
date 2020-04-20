# CubiCal: a radio interferometric calibration suite
# (c) 2017 Rhodes University & Jonathan S. Kenyon
# http://github.com/ratt-ru/CubiCal
# This code is distributed under the terms of GPLv2, see LICENSE.md for details
from __future__ import print_function
from cubical.machines.complex_2x2_machine import Complex2x2Gains
import numpy as np
import numpy.ma as ma
from numpy.ma import masked_array
from cubical.flagging import FL
import cubical.kernels

from cubical.tools import logger

log = logger.getLogger("complex_pol")


class PolarizationGains(Complex2x2Gains):
    """
    This class implements the full complex 2x2 gain machine.
    """

    def __init__(self, label, data_arr, ndir, nmod, chunk_ts, chunk_fs, chunk_label, options):
        """
        Initialises a 2x2 polarization gain machine (leakage and PZD)

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
        super(PolarizationGains, self).__init__(label, data_arr, ndir, nmod,
                                  chunk_ts, chunk_fs, chunk_label, options)

        self._estimate_pzd = options["estimate-pzd"]
        # this will be set to PZD and exp(-i*PZD) once the PZD estimate is done
        self._exp_pzd = self._pzd = None

    @classmethod
    def determine_diagonality(cls, options):
        """Returns true if the machine class, given the options, represents a diagonal gain"""
        return False

    @staticmethod
    def exportable_solutions():
        """ Returns a dictionary of exportable solutions for this machine type. """
        sols = Complex2x2Gains.exportable_solutions()
        sols["pzd"] = (0.0, ("time", "freq"))
        return sols

    def importable_solutions(self, grid0):
        """ Returns a dictionary of importable solutions for this machine type. """
        sols = super(PolarizationGains, self).importable_solutions(grid0)
        sols["pzd"] = dict(**self.interval_grid)
        return sols

    def export_solutions(self):
        """ Saves the solutions to a dict of {label: solutions,grids} items. """
        sols = super(PolarizationGains, self).export_solutions()
        sols["pzd"] = (masked_array(self._pzd if self._pzd is not None else 0.), self.interval_grid)
        return sols

    def import_solutions(self, soldict):
        """
        Loads solutions from a dict.

        Args:
            soldict (dict):
                Contains gains solutions which must be loaded.
        """
        if "pzd" in soldict:
            self._pzd = soldict["pzd"]
            self._exp_pzd = np.exp(-1j * self._pzd)

        super(PolarizationGains, self).import_solutions(soldict)

    #        import ipdb; ipdb.set_trace()

    def precompute_attributes(self, data_arr, model_arr, flags_arr, noise):
        """
        """
        super(PolarizationGains, self).precompute_attributes(data_arr, model_arr, flags_arr, noise)

        if self._pzd is None and self._estimate_pzd:
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
            print("{0}: PZD estimate {1} deg".format(self.chunk_label, pzd * 180 / np.pi), file=log(0))
            self._pzd = pzd
            self._exp_pzd = np.exp(-1j * pzd)

            self.gains[:, :, :, :, 0, 0] = 1
            self.gains[:, :, :, :, 1, 1] = self._exp_pzd[np.newaxis, :, :, np.newaxis]


    def restrict_solution(self, gains):
        """
        Restricts the solution by invoking the inherited restrict_soultion method and applying
        any machine specific restrictions.
        """

        if "xd" in self.update_type:
            if self._estimate_pzd and self.solvable:
                # re-estimate pzd
                mask = self.gflags!=0
                pzd = masked_array(gains[:, :, :, :, 0, 0] / gains[:, :, :, :, 1, 1], mask)
                pzd = np.angle(pzd.sum(axis=(0,3)))
                print("{0}: PZD estimate changes by {1} deg".format(self.chunk_label, (pzd-self._pzd)* 180 / np.pi), file=log(0))
                # import ipdb; ipdb.set_trace()
                self._pzd = pzd
                self._exp_pzd = np.exp(-1j * pzd)

            gains[:, :, :, :, 0, 0] = 1
            gains[:, :, :, :, 1, 1] = self._exp_pzd[np.newaxis, :, :, np.newaxis] if self._exp_pzd is not None else 1

            # px = ma.masked_array(np.angle(gains[:,:,:,:,0,0]), self.gflags)
            # py = ma.masked_array(np.angle(gains[:,:,:,:,1,1]), self.gflags)
            # take phase difference, -pi to pi, and then mean over antennas
            # pzd_exp = ma.exp(1j*(py-px))
            # pzd = ma.angle(pzd_exp.product(axis=-1)) / pzd_exp.count(axis=-1)
            # pzd = (ma.fmod(py - px, 2*np.pi) - np.pi).mean(axis=-1)
            # print("{0}: PZD update is {1} deg".format(self.chunk_label, pzd*180/np.pi), file=log(0))
            # assign to all antennas
            # gains[:,:,:,:,0,0] = 1
            # gains[:,:,:,:,1,1] = np.exp(1j*pzd)[...,np.newaxis]

        super(PolarizationGains, self).restrict_solution(gains)

    @property
    def dof_per_antenna(self):
        # two complex leakages, plus one common PZD across antennas
        if "xd" in self.update_type:
            return 4 + 1./self.n_ant
        else:
            return 8
