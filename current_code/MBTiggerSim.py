import collections
import functools
import types

import numpy as np
import pyrap.tables as pt

import montblanc
import logging
import montblanc.util as mbu
import montblanc.impl.rime.tensorflow.ms.ms_manager as MS

from montblanc.config import RimeSolverConfig as Options
from montblanc.impl.rime.tensorflow.sources import (SourceProvider,
                                                    FitsBeamSourceProvider,
                                                    MSSourceProvider)
from montblanc.impl.rime.tensorflow.sinks import (SinkProvider,
                                                  MSSinkProvider)


class MSSourceProvider(SourceProvider):
    def __init__(self, handler, ntime, nchan):

        self._handler = handler
        self._ms = self._handler.ms
        self._ms_name = self._handler.ms_name
        self._name = "Measurement set '{ms}'".format(ms=self._ms_name)

        self._ntime = ntime
        self._nchan = nchan
        self._nbl = (self._handler.nants*(self._handler.nants - 1))/2
        self._ft = self._handler._first_t
        self._lt = self._handler._last_t
        self._ff = self._handler._first_f
        self._lf = self._handler._last_f
        self._times = self._handler.rtime[self._ft:self._lt]
        self._flags = self._handler.flags[self._ft:self._lt,
                                          self._ff:self._lf, :]
        self._weigh = self._handler.weigh[self._ft:self._lt, :]
        self._uvwco = self._handler.uvwco[self._ft:self._lt, :]


    def name(self):
        return self._name

    def updated_dimensions(self):
        return [('ntime', self._ntime),
                ('nbl', self._nbl),
                ('na', self._handler.nants),
                ('nchan', self._nchan),
                ('nbands', 1),
                ('npol', 4),
                ('npolchan', 4*self._nchan)]

    def frequency(self, context):
        channels = self._handler._chanfr[0, self._ff:self._lf]
        return channels.reshape(context.shape).astype(context.dtype)

    def ref_frequency(self, context):
        # num_chans = self._handler._nchans
        ref_freqs = self._handler._rfreqs

        data = np.hstack((np.repeat(rf, bs) for bs, rf in zip([self._nchan],
                                                              ref_freqs)))
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
            # Inspection confirms that this achieves the same effect as
            # ant_uvw[ti,1:na,:] = ...getcol(UVW, ...).reshape(na-1, -1)
            ant_uvw[ti,1:na,:] = self._uvwco[t*nbl:t*nbl+na-1, :]

        return ant_uvw

    def antenna1(self, context):
        lrow, urow = MS.uvw_row_extents(context)
        antenna1 = self._handler.antea[lrow:urow]

        return antenna1.reshape(context.shape).astype(context.dtype)

    def antenna2(self, context):
        lrow, urow = MS.uvw_row_extents(context)
        antenna2 = self._handler.anteb[lrow:urow]

        return antenna2.reshape(context.shape).astype(context.dtype)

    def parallactic_angles(self, context):
        # Time and antenna extents
        (lt, ut), (la, ua) = context.dim_extents('ntime', 'na')

        return mbu.parallactic_angles(self._handler._phadir,
            self._handler._antpos[la:ua], self._times[lt:ut]).astype(
            context.dtype)

    # def observed_vis(self, context):
    #     lrow, urow = MS.row_extents(context)
    #
    #     data = self._handler.obvis(startrow=lrow, nrow=urow-lrow)
    #
    #     return data.reshape(context.shape).astype(context.dtype)

    def flag(self, context):
        lrow, urow = MS.row_extents(context)

        flag = self._flags[lrow:urow]

        return flag.reshape(context.shape).astype(context.dtype)

    def weight(self, context):
        lrow, urow = MS.row_extents(context)

        weight = self._weigh[lrow:urow]

        weight = np.repeat(weight, self._nchan, 0)

        return weight.reshape(context.shape).astype(context.dtype)

    def __enter__(self):
        return self

    def __exit__(self, etype, evalue, etraceback):
        self.close()

    def __str__(self):
        return self.__class__.__name__


class ArraySinkProvider(SinkProvider):
    def __init__(self, handler, ntime, nchan):
        self._name = "Measurement Set '{ms}'".format(ms=handler.ms_name)
        self._sim_array = np.zeros([ntime, nchan, handler.nants,
                                    handler.nants, 4], dtype=handler.ctype)

    def name(self):
        return self._name

    def model_vis(self, context):

        ntime, nbl, nchan, ncorr = context.data.shape

        a1 = context.input["antenna1"]
        a2 = context.input["antenna2"]

        for t in xrange(ntime):
            self._sim_array[t,:,a1[t,:],a2[t,:],:] = context.data[t,...]
            self._sim_array[t,:,a2[t,:],a1[t,:],:] = context.data[t,...].conj()[...,(0,2,1,3)]

    def __str__(self):
        return self.__class__.__name__



def simulate(srcprov, sinkprov):

    montblanc.log.setLevel(logging.DEBUG)
    [h.setLevel(logging.DEBUG) for h in montblanc.log.handlers]

    slvr_cfg = montblanc.rime_solver_cfg(
        mem_budget=1024*1024*1024,
        dtype='float',
        version=Options.VERSION_TENSORFLOW)

    with montblanc.rime_solver(slvr_cfg) as slvr:
        # Manages measurement sets

        source_provs = []
        # Read problem info from the MS, taking observed visibilities from MODEL_DAT
        source_provs.extend([srcprov])

        # Add a beam when you're ready
        #source_provs.append(FitsBeamSourceProvider('beam_$(corr)_$(reim).fits'))
        #tsp = TiggerSourceProvider("3C147-GdB-spw0+pybdsm.lsm.html",
        # args.msfile)
        #source_provs.append(tsp)

        sink_provs = []
        sink_provs.extend([sinkprov])
        # Dump model visibilities into CORRECTED_DATA
        # sink_provs.append(MSSinkProvider(ms_mgr, 'CORRECTED_DATA'))
        #ntime, nchan, na = ms_mgr._dim_sizes['ntime'], ms_mgr._dim_sizes[
        # 'nchan'], ms_mgr._dim_sizes['na']
        #msp = ModelSinkProvider(ms_mgr, tsp._nclus, ntime, nchan, na)
        #sink_provs.append(msp)

        slvr.solve(source_providers=source_provs, sink_providers=sink_provs)