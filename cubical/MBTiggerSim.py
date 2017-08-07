import collections
import functools
import types

import numpy as np
import pyrap.tables as pt

import montblanc
import logging
import montblanc.util as mbu
import montblanc.impl.rime.tensorflow.ms.ms_manager as MS

from montblanc.impl.rime.tensorflow.sources import (SourceProvider,
                                                    FitsBeamSourceProvider,
                                                    MSSourceProvider)
from montblanc.impl.rime.tensorflow.sinks import (SinkProvider,
                                                  MSSinkProvider)

class MSSourceProvider(SourceProvider):
    def __init__(self, tile, data, sort_ind):

        self._tile = tile
        self._handler = tile.handler
        self._ms = self._handler.ms
        self._ms_name = self._handler.ms_name
        self._name = "Measurement set '{ms}'".format(ms=self._ms_name)

        self._ntime = len(np.unique(self._tile.times))
        self._nchan = self._handler._nchans[0]
        self._nants = self._handler.nants
        self._ncorr = self._handler.ncorr
        self._nbl   = (self._nants*(self._nants - 1))/2
        self._times = self._tile.time_col
        self._antea = self._tile.antea
        self._anteb = self._tile.anteb
        self._ddids = self._tile.ddids
        self._nddid = len(self._tile.ddids)
        self._uvwco = data['uvwco']
        self.sort_ind = sort_ind

    def name(self):
        return self._name

    def updated_dimensions(self):
        return [('ntime', self._ntime),
                ('nbl', self._nbl),
                ('na', self._nants),
                ('nchan', self._nchan*self._nddid),
                ('nbands', self._nddid),
                ('npol', self._ncorr),
                ('npolchan', 4*self._nchan)]

    def frequency(self, context):
        channels = self._handler._chanfr[self._ddids, :]
        return channels.reshape(context.shape).astype(context.dtype)

    def uvw(self, context):
        """ Special case for handling antenna uvw code """

        # Figure out our extents in the time dimension
        # and our global antenna and baseline sizes
        (t_low, t_high) = context.dim_extents('ntime')
        na, nbl = context.dim_global_size('na', 'nbl')

        # We expect to handle all antenna at once
        if context.shape != (t_high - t_low, na, 3):
            raise ValueError("Received an unexpected shape "
                "{s} in (ntime,na,3) antenna reading code".format(s=context.shape))

        # Create per antenna UVW coordinates.
        # u_01 = u_1 - u_0
        # u_02 = u_2 - u_0
        # ...
        # u_0N = u_N - U_0
        # where N = na - 1.

        # Choosing u_0 = 0 we have:
        # u_1 = u_01
        # u_2 = u_02
        # ...
        # u_N = u_0N

        # Then, other baseline values can be derived as
        # u_21 = u_1 - u_2

        # Allocate space for per-antenna UVW, zeroing antenna 0 at each timestep
        ant_uvw = np.empty(shape=context.shape, dtype=context.dtype)
        ant_uvw[:,0,:] = 0

        # Read in uvw[1:na] row at each timestep
        for ti, t in enumerate(xrange(t_low, t_high)):
            # Inspection confirms that this achieves the same effect as
            # ant_uvw[ti,1:na,:] = ...getcol(UVW, ...).reshape(na-1, -1)
            ant_uvw[ti,1:na,:] = self._uvwco[self.sort_ind, ...][t*nbl:t*nbl+na-1, :]

        return ant_uvw

    def antenna1(self, context):
        lrow, urow = MS.uvw_row_extents(context)
        antenna1 = self._antea[self.sort_ind][lrow:urow]

        return antenna1.reshape(context.shape).astype(context.dtype)

    def antenna2(self, context):
        lrow, urow = MS.uvw_row_extents(context)
        antenna2 = self._anteb[self.sort_ind][lrow:urow]

        return antenna2.reshape(context.shape).astype(context.dtype)

    def parallactic_angles(self, context):
        # Time and antenna extents
        (lt, ut), (la, ua) = context.dim_extents('ntime', 'na')

        return mbu.parallactic_angles(np.unique(self._times[self.sort_ind])[lt:ut], self._handler._antpos[la:ua], 
                    self._handler._phadir).reshape(context.shape).astype(context.dtype)

    def __enter__(self):
        return self

    def __exit__(self, etype, evalue, etraceback):
        self.close()

    def __str__(self):
        return self.__class__.__name__


class ColumnSinkProvider(SinkProvider):
    def __init__(self, tile, data, sort_ind):
        self._tile = tile
        self._data = data
        self._handler = tile.handler
        self._name = "Measurement Set '{ms}'".format(ms=self._handler.ms_name)
        self._dir = 0
        self.sort_ind = sort_ind
        self._ddids = self._tile.ddids
        self._nddid = len(self._ddids)

    def name(self):
        return self._name

    def model_vis(self, context):

        _, _, _, ncorr = context.data.shape

        (lt, ut), (lbl, ubl), (lc, uc) = context.dim_extents('ntime', 'nbl', 'nchan')

        lower, upper = MS.row_extents(context, ("ntime", "nbl"))

        ntime, nbl, nchan = context.dim_global_size('ntime', 'nbl', 'nchan')
        rows_per_ddid = ntime*nbl
        chan_per_ddid = nchan/self._nddid

        for ddid_ind in xrange(self._nddid):
            offset = ddid_ind*rows_per_ddid
            lr = lower + offset
            ur = upper + offset
            lc = ddid_ind*chan_per_ddid
            uc = (ddid_ind+1)*chan_per_ddid
            self._data['movis'][self._dir, 0, lr:ur, :, :] = context.data[:,:,lc:uc,:].reshape(-1, chan_per_ddid, ncorr)

    def __str__(self):
        return self.__class__.__name__

_mb_slvr = None

def simulate(src_provs, snk_provs):

    global _mb_slvr

    mblogger = logging.Logger.manager.loggerDict["montblanc"]
    mblogger.propagate = False

    if _mb_slvr is None:
        slvr_cfg = montblanc.rime_solver_cfg(
            mem_budget=4*1024*1024*1024,
            dtype='double',
            polarisation_type='circular')

        _mb_slvr = montblanc.rime_solver(slvr_cfg)
        
    _mb_slvr.solve(source_providers=src_provs, sink_providers=snk_provs)

import atexit

def _shutdown_mb_slvr():
    
    global _mb_slvr
    
    if _mb_slvr is not None:
        _mb_slvr.close()

atexit.register(_shutdown_mb_slvr)

