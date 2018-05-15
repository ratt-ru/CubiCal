# CubiCal: a radio interferometric calibration suite
# (c) 2017 Rhodes University & Jonathan S. Kenyon
# http://github.com/ratt-ru/CubiCal
# This code is distributed under the terms of GPLv2, see LICENSE.md for details
"""
Cython kernels for various generic operations. Common dimensions of arrays are:

+----------------+------+
| Dimension      | Size |
+================+======+
| Direction      |   d  |
+----------------+------+
| Model          |   m  |
+----------------+------+
| Time           |   t  |
+----------------+------+
| Time Intervals |   ti |
+----------------+------+
| Frequency      |   f  |
+----------------+------+
| Freq Intervals |   fi |
+----------------+------+
| Antenna        |   a  |
+----------------+------+
| Correlation    |   c  |
+----------------+------+

"""

import numpy as np
cimport numpy as np
import cython
from cython.parallel import parallel, prange
import cubical.kernels

ctypedef np.complex64_t fcomplex
ctypedef np.complex128_t dcomplex

ctypedef fused complex3264:
    fcomplex
    dcomplex

@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def cycompute_2x2_inverse(complex3264 [:,:,:,:,:,:] x,
                         complex3264 [:,:,:,:,:,:] xinv,
                         np.uint16_t [:,:,:,:] flags,
                         float eps,
                         int flagbit):
    """
    Given an X array of dimensiond (d,t,i,a,2,2), computes inverse of every 2x2 block. Takes flags
    of shape (d,t,f,a) into account, and will flag elements if the inverse is too large.

    Args:
        x (np.complex64 or np.complex128):
            Typed memoryview of X array with dimensions (d, ti, fi, a, c, c)
        xinv (np.complex64 or np.complex128):
            Typed memoryview of output inverse array with dimensions
            (d, ti, fi, a, c, c)
        flags (np.uint16_t):
            Typed memoryview of flag array with dimensions (d, t, f, a)
        eps (float):
            Threshold beneath which the denominator is regarded as too small for inversion.
        flagbit (int):
            The bitflag which will be raised if flagging is required.

    Returns:
        int:
            Number of elements flagged
    """

    cdef int d, t, f, aa, ab = 0
    cdef int n_dir, n_tim, n_fre, n_ant
    cdef complex3264 denom = 0
    cdef int flag_count = 0

    eps = eps**2

    n_dir = x.shape[0]
    n_tim = x.shape[1]
    n_fre = x.shape[2]
    n_ant = x.shape[3]

    cdef int num_threads = cubical.kernels.num_omp_threads
    with nogil, parallel(num_threads=num_threads):
        for aa in prange(n_ant, schedule='static'):
            for t in xrange(n_tim):
                for f in xrange(n_fre):
                    for d in xrange(n_dir):
                        if flags[d,t,f,aa]:

                                xinv[d,t,f,aa,0,0] = 0
                                xinv[d,t,f,aa,1,1] = 0
                                xinv[d,t,f,aa,0,1] = 0
                                xinv[d,t,f,aa,1,0] = 0

                        else:
                            denom = x[d,t,f,aa,0,0] * x[d,t,f,aa,1,1] - \
                                    x[d,t,f,aa,0,1] * x[d,t,f,aa,1,0]

                            if (denom*denom.conjugate()).real<=eps:

                                xinv[d,t,f,aa,0,0] = 0
                                xinv[d,t,f,aa,1,1] = 0
                                xinv[d,t,f,aa,0,1] = 0
                                xinv[d,t,f,aa,1,0] = 0

                                flags[d,t,f,aa] = flagbit
                                flag_count += 1

                            else:

                                xinv[d,t,f,aa,0,0] = x[d,t,f,aa,1,1]/denom
                                xinv[d,t,f,aa,1,1] = x[d,t,f,aa,0,0]/denom
                                xinv[d,t,f,aa,0,1] = -1 * x[d,t,f,aa,0,1]/denom
                                xinv[d,t,f,aa,1,0] = -1 * x[d,t,f,aa,1,0]/denom

    return flag_count

@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def cycompute_diag_inverse(complex3264 [:,:,:,:,:,:] x,
                     complex3264 [:,:,:,:,:,:] xinv,
                     np.uint16_t [:,:,:,:] flags,
                     float eps,
                     int flagbit):
    """
    Like compute_2x2_inv, but the 2x2 blocks of X are assumed to be diagonal
    """

    cdef int d, t, f, aa, ab = 0
    cdef int n_dir, n_tim, n_fre, n_ant
    cdef complex3264 denom = 0
    cdef int flag_count = 0

    eps = eps**2

    n_dir = x.shape[0]
    n_tim = x.shape[1]
    n_fre = x.shape[2]
    n_ant = x.shape[3]

    cdef int num_threads = cubical.kernels.num_omp_threads
    with nogil, parallel(num_threads=num_threads):
        for aa in prange(n_ant, schedule='static'):
            for t in xrange(n_tim):
                for f in xrange(n_fre):
                    for d in xrange(n_dir):
                        xinv[d,t,f,aa,0,1] = xinv[d,t,f,aa,1,0] = 0
                        if flags[d,t,f,aa]:
                            xinv[d,t,f,aa,0,0] = xinv[d,t,f,aa,1,1] = 0
                        else:
                            denom = x[d,t,f,aa,0,0] * x[d,t,f,aa,1,1]
                            if (denom*denom.conjugate()).real<=eps:
                                xinv[d,t,f,aa,0,0] = xinv[d,t,f,aa,1,1] = 0
                                flags[d,t,f,aa] = flagbit
                                flag_count += 1
                            else:
                                xinv[d,t,f,aa,0,0] = 1/x[d,t,f,aa,0,0]
                                xinv[d,t,f,aa,1,1] = 1/x[d,t,f,aa,1,1]

    return flag_count

_half_baselines = _half_baselines_nant = _all_baselines = _all_baselines_nant = None

cdef np.int32_t [:,:] _half_baselines_view
cdef np.int32_t [:,:] _all_baselines_view

def half_baselines(int n_ant):
    """
    Computes an Nx2 array of unique baseline indices, given a number of antennas
    Returns only the p<q baselines.

    This is a convenience object that can be iterated over by prange() and its ilk.
    """
    global _half_baselines
    global _half_baselines_view
    cdef int i
    if n_ant != _half_baselines_nant:
        nbl = n_ant*(n_ant-1)/2
        _half_baselines = np.empty((nbl,2),np.int32)
        _half_baselines_view = _half_baselines
        i = 0
        for p in xrange(n_ant-1):
            for q in xrange(p+1, n_ant):
                _half_baselines_view[i][0] = p
                _half_baselines_view[i][1] = q
                i += 1
    return _half_baselines_view

def all_baselines(int n_ant):
    """
    Computes an Nx2 array of unique baseline indices, given a number of antennas
    Returns the p!=q baselines.
    """
    global _all_baselines
    global _all_baselines_view
    cdef int i
    if n_ant != _all_baselines_nant:
        nbl = n_ant*(n_ant-1)
        _all_baselines = np.empty((nbl,2),np.int32)
        _all_baselines_view = _all_baselines
        i = 0
        for p in xrange(n_ant):
            for q in xrange(n_ant):
                if p != q:
                    _all_baselines_view[i][0] = p
                    _all_baselines_view[i][1] = q
                    i += 1
    return _all_baselines_view

@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def cycompute_chisq(complex3264 [:,:,:,:,:,:,:] resid,np.float64_t [:,:,:] chisq):
    """
    # Compute chi-square over correlations, models, and one antenna axis. Result has shape
    # (n_tim, n_fre, n_ant). We avoid using np.abs by taking a view of the underlying memory.
    # This is substantially faster.
    """

    cdef int d, i, t, f, aa, ab, c1, c2
    cdef int n_mod, n_tim, n_fre, n_ant

    n_mod = resid.shape[0]
    n_tim = resid.shape[1]
    n_fre = resid.shape[2]
    n_ant = resid.shape[3]

    cdef int num_threads = cubical.kernels.num_omp_threads

    with nogil, parallel(num_threads=num_threads):
        for aa in prange(n_ant, schedule='static'):
            for ab in xrange(n_ant):
                for i in xrange(n_mod):
                    for t in xrange(n_tim):
                        for f in xrange(n_fre):
                            for c1 in xrange(2):
                                for c2 in xrange(2):
                                    chisq[t,f,aa] += resid[i,t,f,aa,ab,c1,c2].real**2 + resid[i,t,f,aa,ab,c1,c2].imag**2

@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def cyrebin_vis(fcomplex [:,:,:] vis,  const fcomplex [:,:,:] vis0,
                double   [:,:]   uvw,  const double   [:,:]   uvw0,
                double   [:]     time, const double   [:]     time0,
                int      [:,:,:] flag, const int      [:,:,:] flag0,
                double   [:,:,:,:] weights,const double  [:,:,:,:]  weights0, int num_weights,
                int [:] timeslots,  int [:] anta,  int [:] antb,
                const int [:] timeslots0, const int [:] anta0, const int [:] antb0,
                int rebin_time, int rebin_freq):
    """
    Rebin the input data
    """

    cdef int d, i, t, f, f0, c, w
    cdef int n_row0, n_fre0, n_cor0, n_fre, n_cor, n_ant, max_nrow, ts_in, ts_out, a1, a2, row0, row, nrow_out
    cdef int conjugate
    cdef double ww=1, row_sum_weights

    n_row0  = vis0.shape[0]
    n_fre0  = vis0.shape[1]
    n_cor0  = n_cor = vis0.shape[2]
    n_row   = vis.shape[0]
    n_fre   = vis.shape[1]
    n_ant   = max(max(anta0), max(antb0))+1

    # this is a map from output timeslot and baseline to the output row allocated to it. Starts as -1
    # and allocated when that timeslot/baseline combination actually comes up
    output_row = np.full(((timeslots0[n_row0-1]+1)/rebin_time, n_ant, n_ant), -1, int)

    sum_ww = np.zeros((n_row,n_fre,n_cor), float)    # sum of weights per each output visibility
    sum_rw = np.zeros(n_row, float)                  # sum of weights per each output row
    num_rr = np.zeros(n_row, int)                    # number of input rows in each output row

    nrow_out = 0                                     # number of rows alocated in output

    for row0 in xrange(n_row0):
        ts_in = timeslots0[row0]
        ts_out = ts_in / rebin_time
        a1 = anta0[row0]
        a2 = antb0[row0]
        conjugate = a2<a1
        if conjugate:
            a1, a2 = a2, a1
        # have we already allocated an output row for this timeslot of the rebinned a1, a2 data?
        row = output_row[ts_out, a1, a2]
        if row < 0:
            row = output_row[ts_out, a1, a2] = nrow_out
            nrow_out += 1
            timeslots[row] = ts_out
            anta[row] = a1
            antb[row] = a2
            num_rr[row] += 1

        row_sum_weights = 0  # sum of weights of current _input_ row

        for f0 in xrange(n_fre0):
            f = f0 / rebin_freq
            for c in xrange(n_cor0):
                flag[row, f, c] &= flag0[row0, f0, c]
                if not flag0[row0, f0, c]:
                    # accumulate weights
                    if num_weights:
                        for w in xrange(num_weights):
                            weights[w, row, f, c] += weights0[w, row0, f0, c]
                        ww = weights[w, row, f, c]
                    vis[row, f, c] += ww*vis0[row0, f0, c].conjugate() if conjugate else ww*vis0[row0, f0, c]
                    sum_ww[row, f, c] += ww
                    sum_rw[row] += ww
                    row_sum_weights += ww
        if row_sum_weights:
            for i in xrange(3):
                uvw[row, i] += row_sum_weights*uvw0[row0, i]
        # time centroid is not weighted. This is potentially a source of error?
        time[row] += time0[row0]

    # now normalize by counts and sums of the weights
    for row in xrange(nrow_out):
        if num_rr[row]:
            time[row] /= num_rr[row]
            if sum_rw[row]:
                for i in xrange(3):
                    uvw[row, i] /= sum_rw[row]
                for c in xrange(n_cor0):
                    if sum_ww[row, f, c]:
                        vis[row, f, c] /= sum_ww[row, f, c]

    return nrow_out



