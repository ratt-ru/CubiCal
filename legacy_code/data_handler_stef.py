import pyrap.tables as pt
import numpy as np
import time


def open_ms(msname):
    """
    Convenience function for opening an MS which obviates reopeneing of the MS.
    """

    return pt.table(msname, readonly=False)


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

    return times.astype(np.int64)


def ms2mat(vis, anta, antb, t_slot, n_ant, n_tim):
    """
    Stacks visibilities into a matrix.
    """

    mat = np.zeros([n_tim, n_ant, n_ant],dtype=np.complex128)

    mat[t_slot, anta, antb] = vis[:, 0]
    mat[t_slot, antb, anta] = vis[:, 0].conj()

    for i in range(n_tim):
        np.fill_diagonal(mat[i,:,:], 0)

    return mat

def mat2ms(mat, anta, antb, t_slot, n_ant, n_tim):
    """
    Stacks visibilities into a matrix.
    """

    vis = np.zeros([n_tim*(n_ant**2+n_ant)/2, 1],dtype=np.complex128)

    vis[:, 0] = mat[t_slot, anta, antb]

    return vis

def save_data(data, ms, col_name):

    ms.putcol(col_name, data[:, :, :])


def tst(vis, anta, antb, t_slot, n_ant, n_tim, n_freq=1 ):
    """
    Stacks visibilities into a matrix.
    """

    mat = np.zeros([n_freq, n_tim, n_ant, n_ant],dtype=np.complex128)

    for i in range(n_freq):
        mat[i, t_slot, anta, antb] = vis[:, i]
        mat[i, t_slot, antb, anta] = vis[:, i].conj()

        for j in range(n_tim):
            np.fill_diagonal(mat[i,j,:,:], 0)

    return mat

def tst2(mat, anta, antb, t_slot, n_ant, n_tim, n_freq=1):
    """
    Returns the matrix form to visibilities.
    """

    vis = np.zeros([n_tim*(n_ant**2+n_ant)/2, n_freq, 1],dtype=np.complex128)

    for i in range(n_freq):
        vis[:, i, 0] = mat[i, t_slot, anta, antb]

    return vis
