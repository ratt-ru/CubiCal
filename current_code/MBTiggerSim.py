import collections
import functools
import types

import numpy as np
import pyrap.tables as pt

import montblanc.util as mbu
from montblanc.config import RimeSolverConfig as Options
import montblanc.impl.rime.tensorflow.ms.ms_manager as MS

from montblanc.impl.rime.tensorflow.sources.source_provider import SourceProvider


class MSSourceProvider(SourceProvider):
    def __init__(self, handler):

        self._handler = handler
        self._ms_name = self._handler.ms_name
        self._name = "Measurement set '{ms}'".format(ms=self._ms_name)
        self._times = self._handler.times
        self._ms = handler.ms

        self._anttab = pt.table(self._ms_name + "::ANTENNA")
        self._fldtab = pt.table(self._ms_name + "::FIELD")
        self._spwtab = pt.table(self._ms_name + "::SPECTRAL_WINDOW")

        self._antpos = self._anttab.getcol("POSITION")
        # Add specific field here.
        self._phase_dir = self._fldtab.getcol("PHASE_DIR")[0][0]

    def name(self):
        return self._name

    def updated_dimensions(self):
        # UPDATE_DIMENSIONS = []
        # Defer to manager's method
        return [('ntime', len(np.unique(self._times))),
                ('nbl', 91),
                ('na', 14),
                ('nchan', 64),
                ('nbands', 1),
                ('npol', 4),
                ('npolchan', 256),
                ('nvis', self._handler.obvis.shape[0])]

    def frequency(self, context):
        channels = self._spwtab.getcol("CHAN_FREQ")
        return channels.reshape(context.shape).astype(context.dtype)

    def ref_frequency(self, context):
        num_chans = self._spwtab.getcol("NUM_CHAN")
        ref_freqs = self._spwtab.getcol("REF_FREQUENCY")

        data = np.hstack((np.repeat(rf, bs) for bs, rf in zip(num_chans, ref_freqs)))
        return data.reshape(context.shape).astype(context.dtype)

    def uvw(self, context):
        """ Special case for handling antenna uvw code """

        # Figure out our extents in the time dimension
        # and our global antenna and baseline sizes
        (t_low, t_high) = context.dim_extents('ntime')
        na, nbl = context.dim_global_size('na', 'nbl')

        # We expect to handle all antenna at once
        if context.shape != (t_high - t_low, na, 3):
            raise ValueError("Received an unexpected shape "
                "{s} in (ntime,na,3) antenna reading code".format(
                    s=context.shape))

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


        # print t_low, t_high
        # Read in uvw[1:na] row at each timestep
        for ti, t in enumerate(xrange(t_low, t_high)):
            print self._ms.nrows(), t*nbl, t*nbl+na-1
            # Inspection confirms that this achieves the same effect as
            # ant_uvw[ti,1:na,:] = ...getcol(UVW, ...).reshape(na-1, -1)
            self._ms.getcolnp("UVW", ant_uvw[ti,1:na,:], startrow=t*nbl,
                              nrow=na-1)

        return ant_uvw

    def antenna1(self, context):
        lrow, urow = MS.uvw_row_extents(context)
        antenna1 = self._ms.getcol("ANTENNA1", startrow=lrow, nrow=urow-lrow)

        return antenna1.reshape(context.shape).astype(context.dtype)

    def antenna2(self, context):
        lrow, urow = MS.uvw_row_extents(context)
        antenna2 = self._ms.getcol("ANTENNA2", startrow=lrow, nrow=urow-lrow)

        return antenna2.reshape(context.shape).astype(context.dtype)

    def parallactic_angles(self, context):
        # Time and antenna extents
        (lt, ut), (la, ua) = context.dim_extents('ntime', 'na')

        return mbu.parallactic_angles(self._phase_dir,
            self._antpos[la:ua], self._times[lt:ut]).astype(context.dtype)

    def observed_vis(self, context):
        lrow, urow = MS.row_extents(context)

        data = self._ms.getcol("DATA", startrow=lrow, nrow=urow-lrow)

        return data.reshape(context.shape).astype(context.dtype)

    def flag(self, context):
        lrow, urow = MS.row_extents(context)

        flag = self._ms.getcol("FLAG", startrow=lrow, nrow=urow-lrow)

        return flag.reshape(context.shape).astype(context.dtype)

    def weight(self, context):
        lrow, urow = MS.row_extents(context)

        weight = self._ms.getcol("WEIGHT", startrow=lrow, nrow=urow-lrow)

        # WEIGHT is applied across all channels
        # TODO: Need to fix channels per band here.
        weight = np.repeat(weight, 64, 0)
        return weight.reshape(context.shape).astype(context.dtype)


    def __enter__(self):
        return self

    def __exit__(self, etype, evalue, etraceback):
        self.close()

    def __str__(self):
        return self.__class__.__name__