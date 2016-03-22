import pyrap.tables as pt
import numpy as np
import time


def open_ms(msname):
    """
    Convenience function for opening an MS which obviates reopeneing of the MS.
    """

    return pt.table(msname)


def get_data(ms, datacol):
    """
    Convenience function for interfacing with measurement sets. PLACEHOLDER:
    Will need to be built up to handle multiple data accesses efficiently.
    """

    return ms.getcol(datacol)


def get_stokes(vis, stokes="I", nchan=1):
    """
    Convenience function for the computation of stokes parameters from the
    stored brightness matrices.
    """

    stokes = stokes.upper()

    if stokes == "I":
        return 0.5*(vis[:, :nchan, 0] + vis[:, :nchan, 3])
    elif stokes == "Q":
        return 0.5*(vis[:, :nchan, 0] - vis[:, :nchan, 3])
    elif stokes == "U":
        return 0.5*(vis[:, :nchan, 1] + vis[:, :nchan, 2])
    elif stokes == "V":
        return 0.5j*(vis[:, :nchan, 1] - vis[:, :nchan, 2])
    else:
        raise ValueError("Stokes parameter must be one of IQUV.")


def times_to_ind(times):
    """
    Converts time values into an array of indices.
    """

    for i,j in enumerate(np.unique(times)):
        times[times==j] = i

    return times


def create_mat(vis, anta, antb, n_ant):
    """
    Stacks visibilities into a matrix.
    """

    mat = np.zeros([n_ant,n_ant],dtype=np.complex128)

    mat[anta, antb] = vis[:, 0]
    mat[antb, anta] = vis[:, 0].conj()
    np.fill_diagonal(mat, 0)

    return mat


# MS = open_ms("~/MeasurementSets/WESTERBORK.MS")
# D = get_data(MS, "DATA")
# D = get_stokes(D)
#
# M = get_data(MS, "MODEL_DATA")
# M = get_stokes(M)
#
# UVW = get_data(MS, "UVW")[:, :2]
#
# times = get_data(MS, "TIME")
#
# t_ind = times_to_ind(times)
#
# ANTA = get_data(MS, "ANTENNA1")
# ANTB = get_data(MS, "ANTENNA2")
#
# t0 = time.time()
# obs_vis = create_mat(D[t_ind==0], ANTA[t_ind==0], ANTB[t_ind==0])
# print time.time() - t0
# mod_vis = create_mat(M[t_ind==0], ANTA[t_ind==0], ANTB[t_ind==0])