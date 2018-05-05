# CubiCal: a radio interferometric calibration suite
# (c) 2017 Rhodes University & Jonathan S. Kenyon
# http://github.com/ratt-ru/CubiCal
# This code is distributed under the terms of GPLv2, see LICENSE.md for details
from abc import ABCMeta, abstractmethod
import numpy as np
from cubical.flagging import FL
from cubical.machines.interval_gain_machine import PerIntervalGains

class ParameterisedGains(PerIntervalGains):
    """
    This is a base class for all gain solution machines that use full-resolution
    gain arrays derived from some other parameters.
    """

    def init_gains(self):
        """
        Construct gain and flag arrays at full-resolution. Parameterisation of gains may not be at
        full resolution.
        """
        self.gain_intervals = 1, 1
        self.gain_grid = self.data_grid
        self.gain_shape = [self.n_dir,self.n_tim,self.n_fre,self.n_ant,self.n_cor,self.n_cor]
        self.gains = np.empty(self.gain_shape, dtype=self.dtype)
        self.gains[:] = np.eye(self.n_cor)
        self.gflags = np.zeros(self.gain_shape[:-2],FL.dtype)

        # Function used to unpack the gains or flags into full time/freq resolution.
        self._gainres_to_fullres  = self.copy_or_identity

        # function used to unpack interval resolution to gain resolution.
        self._interval_to_gainres = self.unpack_intervals

