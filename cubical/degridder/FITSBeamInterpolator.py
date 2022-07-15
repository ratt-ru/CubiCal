# CubiCal: a radio interferometric calibration suite
# (c) 2017 Rhodes University & Jonathan S. Kenyon
# http://github.com/ratt-ru/CubiCal
# This code is distributed under the terms of GPLv2, see LICENSE.md for details
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from multiprocessing.sharedctypes import Value

from DDFacet.compatibility import range
import DDFacet.Data.ClassFITSBeam as DDFacetFITSBeam
import DDFacet.Data.ClassMS as DDFacetClassMS
import numpy
import os
import os.path
import sys
import pyrap.tables
import numpy as np

from cubical.tools import logger, ModColor
log = logger.getLogger("FITSBeamInterpolator")

dm = pyrap.measures.measures()
dq = pyrap.quanta

# This a list of the Stokes enums (as defined in casacore header measures/Stokes.h)
# These are referenced by the CORR_TYPE column of the MS POLARIZATION subtable.
# E.g. 5,6,7,8 corresponds to RR,RL,LR,LL
MS_STOKES_ENUMS = DDFacetFITSBeam.MS_STOKES_ENUMS
# set of circular correlations
CIRCULAR_CORRS = DDFacetFITSBeam.CIRCULAR_CORRS
# set of linear correlations
LINEAR_CORRS = DDFacetFITSBeam.LINEAR_CORRS

class DDFacetMSLite:
    def __init__ (self,
                  field_centre,
                  chunk_ctr_frequencies,
                  chunk_channel_widths,
                  correlation_ids,
                  station_positions,
                  station_names):
        """
            Standin object for the ClassMS visibility reader of DDFacet
            field_centre: radian ra dec coordinate of the original field phase centre
            chunk_ctr_frequencies: center channel frequencies (Hz) of chunk to predict E terms for
            chunk_channel_widths: channel widths of chunk channels
            correlation_ids: hand labels per correlation term as defined in casacore Stokes.h
            station_positions: ECEF coordinates for stations (na, 3) array
            station_names: Station names
        """
        self.ChanFreq = chunk_ctr_frequencies
        self.OriginalRadec = field_centre
        self.ChanWidth = chunk_channel_widths
        self.CorrelationNames = list(map(lambda x: MS_STOKES_ENUMS[x], 
                                         sorted(correlation_ids)))
        self.StationNames = station_names
        self.na = len(station_names)
        self.StationPos = station_positions
        self.rarad, self.decrad = self.OriginalRadec
        if len(self.StationNames) != len(self.StationPos):
            raise ValueError("StationNames should be the length of StationPos")
        
    def radec2lm_scalar(self, *args, **kwargs):
        return DDFacetClassMS.ClassMS.radec2lm_scalar(self, *args, **kwargs)
        
class FITSBeamInterpolator (object):
    def __init__ (self, 
                  field_centre,
                  chunk_ctr_frequencies,
                  chunk_channel_widths,
                  correlation_ids,
                  station_positions,
                  station_names,
                  opts):
        """
            Proxy object for DDFacet.ClassFITSBeam
            field_centre: radian ra dec coordinate of the original field phase centre
            chunk_ctr_frequencies: center channel frequencies (Hz) of chunk to predict E terms for
            chunk_channel_widths: channel widths of chunk channels
            correlation_ids: hand labels per correlation term as defined in casacore Stokes.h
            station_positions: ECEF coordinates for stations (na, 3) array
            station_names: Station names
            opts: beam opts as defined in Parset
        """
        self.__ms = DDFacetMSLite(field_centre,
                                  chunk_ctr_frequencies,
                                  chunk_channel_widths,
                                  correlation_ids,
                                  station_positions,
                                  station_names)
        self.__opts = opts
        self.__vbinterpolator = DDFacetFITSBeam.ClassFITSBeam(self.__ms, self.__opts)
        for method in ('evaluateBeam', 'getFreqDomains', 'getBeamSampleTimes'):
            setattr(self, method, getattr(self.__vbinterpolator, method))



