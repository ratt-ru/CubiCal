# CubiCal: a radio interferometric calibration suite
# (c) 2017 Rhodes University & Jonathan S. Kenyon
# http://github.com/ratt-ru/CubiCal
# This code is distributed under the terms of GPLv2, see LICENSE.md for details

import numpy as np
import sys

def uniquify(values):
    """
    Helper function. Given a vector of values (e.g. times), finds the set of unique values,
    and computes a vector of indices (e.g. timeslots) of the same size, as well as rmap: a map from the
    unique values to their index numbers.

    Returns tuple of indices, unique_values, rmap
    """
    uniq = np.array(sorted(set(values)))
    rmap = {x: i for i, x in enumerate(uniq)}
    # apply this map to the time column to construct a timestamp column
    indices  = np.fromiter(map(rmap.__getitem__, values), int)
    return indices, uniq, rmap

# Try to import montblanc: if not successful, remember error for later.
def import_montblanc():
    """
    Tries to import montblanc. Returns tuple of montblanc_module, None on success, or
    None, exc_info on error
    """
    try:
        import montblanc
        # all of these potentially fall over if Montblanc is the wrong version or something, so moving them here
        # for now
        from MBTiggerSim import simulate, MSSourceProvider, ColumnSinkProvider
        from TiggerSourceProvider import TiggerSourceProvider
        from montblanc.impl.rime.tensorflow.sources import CachedSourceProvider, FitsBeamSourceProvider
        return montblanc, None
    except:
        return None, sys.exc_info()


