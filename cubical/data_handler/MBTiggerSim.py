# CubiCal: a radio interferometric calibration suite
# (c) 2017 Rhodes University & Jonathan S. Kenyon
# http://github.com/ratt-ru/CubiCal
# This code is distributed under the terms of GPLv2, see LICENSE.md for details
"""
Handles the interface between measurement sets, CubiCal and Montblanc.
"""
from __future__ import print_function
from builtins import range
import numpy as np
import montblanc
import montblanc.util as mbu
import montblanc.impl.rime.tensorflow.ms.ms_manager as MS

from montblanc.impl.rime.tensorflow.sources import SourceProvider
from montblanc.impl.rime.tensorflow.sinks import SinkProvider

import datetime as dt
import pyrap.quanta as pq

from cubical.tools import logger, ModColor
log = logger.getLogger("MBSourceProvider")

class MSSourceProvider(SourceProvider):
    """
    Handles interface between CubiCal tiles and Montblanc simulation.
    """

    def __init__(self, tile, time_col, antea, anteb, ddid_col, uvw, freqs, sort_ind, nrows, do_pa_rotation=True):
        """
        Initialises this source provider.

        Args:
            tile (:obj:`~cubical.data_handler.Tile`):
                Tile object containing information about current data selection.
            uvw (np.darray):
                (n_row, 3) array of UVW coordinates.
            freqs (np.darray):
                (n_ddid,n_chan) array of frequencies.
            sort_ind (np.ndarray):
                Indices which will produce sorted data. Montblanc expects data to be ordered in a specific way.
            nrows (int):
                Number of rows in the UNPADDED data. This is necessary for the revised uvw code.

        """

        self._ms = tile.dh.ms
        self._ms_name = tile.dh.ms_name
        self._name = "Measurement set '{ms}'".format(ms=self._ms_name)

        self._antpos = tile.dh.antpos
        self._phadir = tile.dh.phadir

        self._ntime = len(np.unique(time_col))
        self._nchan = freqs.shape[1]
        self._nants = tile.dh.nants
        self._ncorr = tile.dh.ncorr
        self._nbl   = (self._nants*(self._nants - 1))/2
        self._freqs = freqs
        self._times = time_col
        self._antea = antea
        self._anteb = anteb
        self._ddids = ddid_col
        self._nddid = freqs.shape[0]
        self._uvwco = uvw                  #  data['uvwco']
        self._nrows = nrows
        self.sort_ind = sort_ind
        self.do_pa_rotation = do_pa_rotation

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
        return self._freqs.reshape(context.shape).astype(context.dtype)

    def uvw(self, context):
        """ Provides Montblanc with an array of uvw coordinates. """

        # Figure out our extents in the time dimension and our global antenna and baseline sizes.

        (t_low, t_high) = context.dim_extents('ntime')

        # Figure out chunks in time (may be repetitious, but needed an easy fix).

        _, counts = np.unique(self._times[:self._nrows], return_counts=True)

        chunks = np.asarray(counts)

        # Compute per antenna uvw coordinates. Data must be ordered by time.
        # Per antenna uvw coordinates fail on data where time!=time_centroid.

        ant_uvw = mbu.antenna_uvw(self._uvwco[:self._nrows],
                                  self._antea[:self._nrows],
                                  self._anteb[:self._nrows],
                                  chunks,
                                  self._nants,
                                  check_missing=False,
                                  check_decomposition=False,
                                  max_err=100)

        return ant_uvw[t_low:t_high, ...].astype(context.dtype)

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
        if not self.do_pa_rotation:
            return np.zeros(context.shape, dtype=context.dtype)

        def __mjd2dt(utc_timestamp):
            """
            Converts array of UTC timestamps to list of datetime objects for human readable printing
            """
            return [dt.datetime.utcfromtimestamp(pq.quantity(t, "s").to_unix_time()) for t in utc_timestamp]
        utc_times = np.unique(self._times[self.sort_ind])[lt:ut]
        dt_start = __mjd2dt([np.min(utc_times)])[0].strftime('%Y/%m/%d %H:%M:%S')
        dt_end = __mjd2dt([np.max(utc_times)])[0].strftime('%Y/%m/%d %H:%M:%S')
        log(2).print("Computing parallactic angles for times between %s and %s UTC" % (dt_start, dt_end))
        return mbu.parallactic_angles(
                        np.unique(self._times[self.sort_ind])[lt:ut],
                        self._antpos[la:ua],
                        self._phadir).reshape(context.shape).astype(context.dtype)

    def feed_angles(self, context):
        """ Provides Montblanc with an array of feed angles. """

        (la, ua) = context.dim_extents('na')
        # TODO(osmirnov)
        # Please fill me in
        return np.zeros(ua-la, dtype=context.dtype)

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
    def __init__(self, dh, freqshape, model, sort_ind):
        """
        Initialises this sink provider.

        Args:
            dh (:obj:`~cubical.data_handler.MSDataHandler`):
                Data handler object.
            freqshape (tuple):
                Shape of frequency array, i.e. (nddid, nchan)
            model (np.ndarray):
                Array of model visibilities into which output will be written.
            sort_ind (np.ndarray):
                Indices which will produce sorted data. Montblanc expects adata to be ordered.

        """

        self._model = model
        self._ncorr = dh.ncorr
        self._name = "Measurement Set '{ms}'".format(ms=dh.ms_name)
        self._dir = 0
        self.sort_ind = sort_ind
        self._nddid, self._chan_per_ddid = freqshape

    def name(self):
        """ Returns name of associated sink provider. """

        return self._name

    def set_direction(self, idir):
        """Sets current direction being simulated.

        Args:
            idir (int):
                Direction number, from 0 to n_dir-1
        """
        self._dir = idir

    def model_vis(self, context):
        """ Tells Montblanc how to handle the model visibility output. """

        (lt, ut), (lbl, ubl), (lc, uc) = context.dim_extents('ntime', 'nbl', 'nchan')

        lower, upper = MS.row_extents(context, ("ntime", "nbl"))

        ntime, nbl, nchan = context.dim_global_size('ntime', 'nbl', 'nchan')
        rows_per_ddid = ntime*nbl

        if self._ncorr == 1:
            sel = 0
        elif self._ncorr == 2:
            sel = slice(None, None, 3)
        else:
            sel = slice(None)

        for ddid_ind in range(self._nddid):
            offset = ddid_ind*rows_per_ddid
            lr = lower + offset
            ur = upper + offset
            lc = ddid_ind*self._chan_per_ddid
            uc = (ddid_ind+1)*self._chan_per_ddid
            self._model[self._dir, 0, lr:ur, :, :] = \
                    context.data[:,:,lc:uc,sel].reshape(-1, self._chan_per_ddid, self._ncorr)

    def __str__(self):
        return self.__class__.__name__

_mb_slvr = None

def simulate(src_provs, snk_provs, polarisation_type, opts):
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

    if _mb_slvr is None:
        slvr_cfg = montblanc.rime_solver_cfg(
            mem_budget=opts["mem-budget"]*1024*1024,
            dtype=opts["dtype"],
            polarisation_type=polarisation_type,
            device_type=opts["device-type"])

        _mb_slvr = montblanc.rime_solver(slvr_cfg)

    _mb_slvr.solve(source_providers=src_provs, sink_providers=snk_provs)

import atexit

def _shutdown_mb_slvr():

    global _mb_slvr

    if _mb_slvr is not None:
        _mb_slvr.close()

atexit.register(_shutdown_mb_slvr)

