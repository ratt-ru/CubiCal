import logging
import numpy as np

import montblanc
import montblanc.util as mbu

from montblanc.config import RimeSolverConfig as Options

from montblanc.impl.rime.tensorflow.ms import MeasurementSetManager
from montblanc.impl.rime.tensorflow.sources import (SourceProvider,
    FitsBeamSourceProvider,
    MSSourceProvider)
from montblanc.impl.rime.tensorflow.sinks import (SinkProvider,
    MSSinkProvider)

import MBTiggerSim as mbt
from ReadModelHandler import *


if __name__ == '__main__':
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='RIME MS test script')
    parser.add_argument('msfile', help='Measurement Set File')
    parser.add_argument('-b', '--beam',
        type=str, default='', help='Base beam filename')
    parser.add_argument('-np','--npsrc',dest='npsrc',
        type=int, default=10, help='Number of Point Sources')
    parser.add_argument('-ac','--auto-correlations',dest='auto_correlations',
        type=lambda v: v.lower() in ("yes", "true", "t", "1"),
        choices=[True, False], default=False,
        help='Handle auto-correlations')
    parser.add_argument('-v','--version',dest='version', type=str,
        default=Options.VERSION_TENSORFLOW,
        choices=[Options.VERSION_TENSORFLOW],
        help='RIME Pipeline Version.')

    args = parser.parse_args(sys.argv[1:])

    # Set the logging level
    montblanc.log.setLevel(logging.DEBUG)
    [h.setLevel(logging.DEBUG) for h in montblanc.log.handlers]

    slvr_cfg = montblanc.rime_solver_cfg(mem_budget=1024*1024*1024,
                                         dtype='single',
                                         version=args.version)

    with montblanc.rime_solver(slvr_cfg) as slvr:
        # Manages measurement sets
        ms_mgr = ReadModelHandler(args.msfile)
        ms_mgr.mass_fetch()

        source_provs = []
        # Read problem info from the MS, taking observed visibilities from MODEL_DAT
        source_provs.append(mbt.MSSourceProvider(ms_mgr))

        # Add a beam when you're ready
        #source_provs.append(FitsBeamSourceProvider('beam_$(corr)_$(reim).fits'))
        #tsp = TiggerSourceProvider("3C147-GdB-spw0+pybdsm.lsm.html",
        # args.msfile)
        #source_provs.append(tsp)

        sink_provs = []
        # Dump model visibilities into CORRECTED_DATA
        # sink_provs.append(MSSinkProvider(ms_mgr, 'CORRECTED_DATA'))
        #ntime, nchan, na = ms_mgr._dim_sizes['ntime'], ms_mgr._dim_sizes[
        # 'nchan'], ms_mgr._dim_sizes['na']
        #msp = ModelSinkProvider(ms_mgr, tsp._nclus, ntime, nchan, na)
        #sink_provs.append(msp)

        slvr.solve(source_providers=source_provs, sink_providers=sink_provs)