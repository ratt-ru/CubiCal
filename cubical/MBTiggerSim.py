# CubiCal: a radio interferometric calibration suite
# (c) 2017 Rhodes University & Jonathan S. Kenyon
# http://github.com/ratt-ru/CubiCal
# This code is distributed under the terms of GPLv2, see LICENSE.md for details
"""
Handles the interface between measurement sets, CubiCal and Montblanc.
"""

import collections
import functools
import types

import numpy as np
import pyrap.tables as pt

import montblanc
import logging
import montblanc.util as mbu
import montblanc.impl.rime.tensorflow.ms.ms_manager as MS

from montblanc.impl.rime.tensorflow.sources import SourceProvider
from montblanc.impl.rime.tensorflow.sinks import SinkProvider

class MSSourceProvider(SourceProvider):
    """
    Handles interface between CubiCal tiles and Montblanc simulation.
    """
    def __init__(self, tile, data, sort_ind):
        """
        Initialises this source provider.

        Args:
            tile (:obj:`~cubical.data_handler.Tile`):
                Tile object containing information about current data selection.
            data (:obj:`~cubical.tools.shared_dict.SharedDict`):
                Shared dictionary containing measurement set data.
            sort_ind (np.ndarray):
                Indices which will produce sorted data. Montblanc expects adata to be ordered.

        """

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
        """ Returns name of associated source provider. """
        
        return self._name

    def updated_dimensions(self):
        """ Inform Montblanc of the dimensions assosciated with this source provider. """

        return [('ntime', self._ntime),
                ('nbl', self._nbl),
                ('na', self._nants),
                ('nchan', self._nchan*self._nddid),
                ('nbands', self._nddid),
                ('npol', 4),
                ('npolchan', 4*self._nchan)]

    def frequency(self, context):
        """ Provides Montblanc with an array of frequencies. """

        channels = self._handler._chanfr[self._ddids, :]
        return channels.reshape(context.shape).astype(context.dtype)

    def uvw(self, context):
        """ Provides Montblanc with an array of uvw coordinates. """

        # Figure out our extents in the time dimension and our global antenna and baseline sizes.

        (t_low, t_high) = context.dim_extents('ntime')
        na, nbl = context.dim_global_size('na', 'nbl')

        # We expect to handle all antenna at once.

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

        # Allocate space for per-antenna UVW, zeroing antenna 0 at each timestep.

        ant_uvw = np.empty(shape=context.shape, dtype=context.dtype)
        ant_uvw[:,0,:] = 0

        # Read in uvw[1:na] row at each timestep.

        for ti, t in enumerate(xrange(t_low, t_high)):
            # Inspection confirms that this achieves the same effect as
            # ant_uvw[ti,1:na,:] = ...getcol(UVW, ...).reshape(na-1, -1)
            ant_uvw[ti,1:na,:] = self._uvwco[self.sort_ind, ...][t*nbl:t*nbl+na-1, :]

        return ant_uvw

    def antenna1(self, context):
        """ Provides Montblanc with an array of antenna1 values. """

        lrow, urow = MS.uvw_row_extents(context)
        antenna1 = self._antea[self.sort_ind][lrow:urow]

        return antenna1.reshape(context.shape).astype(context.dtype)

    def antenna2(self, context):
        """ Provides Montblanc with an array of antenna2 values. """

        lrow, urow = MS.uvw_row_extents(context)
        antenna2 = self._anteb[self.sort_ind][lrow:urow]

        return antenna2.reshape(context.shape).astype(context.dtype)

    def parallactic_angles(self, context):
        """ Provides Montblanc with an array of parallactic angles. """

        # Time and antenna extents
        (lt, ut), (la, ua) = context.dim_extents('ntime', 'na')

        return mbu.parallactic_angles(
                    np.unique(self._times[self.sort_ind])[lt:ut], self._handler._antpos[la:ua], 
                    self._handler._phadir).reshape(context.shape).astype(context.dtype)

    def __enter__(self):
        return self

    def __exit__(self, etype, evalue, etraceback):
        self.close()

    def __str__(self):
        return self.__class__.__name__


class ColumnSinkProvider(SinkProvider):
    """
    Handles Montblanc output and makes it consistent with the measurement set.
    """

    def __init__(self, tile, data, sort_ind):
        """
        Initialises this sink provider.

        Args:
            tile (:obj:`~cubical.data_handler.Tile`):
                Tile object containing information about current data selection.
            data (:obj:`~cubical.tools.shared_dict.SharedDict`):
                Shared dictionary containing measurement set data.
            sort_ind (np.ndarray):
                Indices which will produce sorted data. Montblanc expects adata to be ordered.

        """

        self._tile = tile
        self._data = data
        self._handler = tile.handler
        self._ncorr = self._handler.ncorr
        self._name = "Measurement Set '{ms}'".format(ms=self._handler.ms_name)
        self._dir = 0
        self.sort_ind = sort_ind
        self._ddids = self._tile.ddids
        self._nddid = len(self._ddids)

    def name(self):
        """ Returns name of associated sink provider. """

        return self._name

    def model_vis(self, context):
        """ Tells Montblanc how to handle the model visibility output. """

        (lt, ut), (lbl, ubl), (lc, uc) = context.dim_extents('ntime', 'nbl', 'nchan')

        lower, upper = MS.row_extents(context, ("ntime", "nbl"))

        ntime, nbl, nchan = context.dim_global_size('ntime', 'nbl', 'nchan')
        rows_per_ddid = ntime*nbl
        chan_per_ddid = nchan/self._nddid

        if self._ncorr == 1:
            sel = 0
        elif self._ncorr == 2:
            sel = slice(None, None, 3)
        else:
            sel = slice(None)

        for ddid_ind in xrange(self._nddid):
            offset = ddid_ind*rows_per_ddid
            lr = lower + offset
            ur = upper + offset
            lc = ddid_ind*chan_per_ddid
            uc = (ddid_ind+1)*chan_per_ddid
            self._data['movis'][self._dir, 0, lr:ur, :, :] = \
                    context.data[:,:,lc:uc,sel].reshape(-1, chan_per_ddid, self._ncorr)

    def __str__(self):
        return self.__class__.__name__

_mb_slvr = None

def simulate(src_provs, snk_provs, opts):
    """
    Convenience function which creates and executes a Montblanc solver for the given source and 
    sink providers.

    Args:
        src_provs (list): 
            List of :obj:`~montblanc.impl.rime.tensorflow.sources.SourceProvider` objects. See
            Montblanc's documentation.
        snk_provs (list):
            List of :obj:`~montblanc.impl.rime.tensorflow.sinks.SinkProvider` objects. See
            Montblanc's documentation. 
        opts (dict):
            Montblanc simulation options (see [montblanc] section in DefaultParset.cfg).
    """

    global _mb_slvr

    mblogger = logging.Logger.manager.loggerDict["montblanc"]
    mblogger.propagate = False

    if _mb_slvr is None:
        slvr_cfg = montblanc.rime_solver_cfg(
            mem_budget=opts["mem-budget"]*1024*1024,
            dtype=opts["dtype"],
            polarisation_type=opts["feed-type"],
            device_type=opts["device-type"])

        _mb_slvr = montblanc.rime_solver(slvr_cfg)
        
    _mb_slvr.solve(source_providers=src_provs, sink_providers=snk_provs)

import atexit

def _shutdown_mb_slvr():
    
    global _mb_slvr
    
    if _mb_slvr is not None:
        _mb_slvr.close()

atexit.register(_shutdown_mb_slvr)

