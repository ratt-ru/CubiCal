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

ctypedef fused float3264:
    np.float32_t
    np.float64_t

# datatype of visibility weights
ctypedef np.float32_t wfloat

ctypedef np.uint16_t flag_t

ctypedef np.int64_t index_t


@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def rebin_index_columns(double [:] time,const double [:] time0,
             np.int64_t [:] antea, const np.int64_t [:] antea0,
             np.int64_t [:] anteb, const np.int64_t [:] anteb0,
             np.int64_t [:] ddid_col, const np.int64_t [:] ddid_col0,
             const index_t [:] row_map):
    """
    Rebins indexing columns (time, ant1/2, ddid), using the given row map.
    time should be a zero-filled output column of the expected length (row_map[-1]+1)
    anteab, anteb, ddid_col should be empty output columns of the expected length
    """
    cdef int row_0, n_row0, n_row, row

    n_row0 = time0.shape[0]
    n_row  = time.shape[0]

    counts = np.zeros(n_row, np.int64)
    cdef np.int64_t [:] vcount = counts

    for row0 in xrange(n_row0):
        row = abs(row_map[row0])
        vcount[row] += 1
        time[row] += time0[row0]
        antea[row] = antea0[row0]
        anteb[row] = anteb0[row0]
        ddid_col[row] = ddid_col0[row0]

    for row in xrange(n_row):
        if vcount[row]:
            time[row] /= vcount[row]




@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def rebin_vis(fcomplex [:,:,:] vis,  const fcomplex [:,:,:] vis0,
              double   [:,:]   uvw,  const double   [:,:]   uvw0,
              flag_t   [:,:,:] flag, const flag_t   [:,:,:] flag0,
              wfloat [:,:,:,:] weights,const wfloat  [:,:,:,:]  weights0, int num_weights,
              const index_t [:] rebin_row_map, const index_t [:] rebin_chan_map):
    """
    Rebin the input data
    """

    cdef int d, i, t, f, f0, c, w
    cdef int n_row0, n_fre0, n_cor0, n_fre, n_cor, n_ant, max_nrow, ts_in, ts_out, a1, a2, row0, row, nrow_out
    cdef int conjugate
    cdef wfloat ww=1, row_sum_weights

    n_row0, n_fre0, n_cor0 = vis0.shape[0], vis0.shape[1], vis0.shape[2]
    n_row, n_fre, n_cor = vis.shape[0], vis.shape[1], n_cor0

    sum_ww0 = np.zeros((n_row,n_fre,n_cor), np.float32)    # sum of weights per each output visibility
    sum_rw0 = np.zeros(n_row, np.float32)                     # sum of weights per each output row

    cdef np.float32_t [:,:,:] sum_ww = sum_ww0
    cdef np.float32_t [:] sum_rw = sum_rw0

    for row0 in xrange(n_row0):
        row = rebin_row_map[row0]
        conjugate = row<0
        if conjugate:
            row = -row

        row_sum_weights = 0  # sum of weights of current _input_ row

        for f0 in xrange(n_fre0):
            f = rebin_chan_map[f0]
            for c in xrange(n_cor0):
                flag[row, f, c] &= flag0[row0, f0, c]
                if not flag0[row0, f0, c]:
                    # accumulate weights
                    if num_weights:
                        for w in xrange(num_weights):
                            weights[w, row, f, c] += weights0[w, row0, f0, c]
                        ww = weights0[0, row0, f0, c]
                    vis[row, f, c] += ww*vis0[row0, f0, c].conjugate() if conjugate else ww*vis0[row0, f0, c]
                    sum_ww[row, f, c] += ww
                    sum_rw[row] += ww
                    row_sum_weights += ww
        if row_sum_weights:
            for i in xrange(3):
                uvw[row, i] += row_sum_weights*uvw0[row0, i]

    # now normalize by counts and sums of the weights
    for row in xrange(n_row):
        if sum_rw[row]:
            for i in xrange(3):
                uvw[row, i] /= sum_rw[row]
            for f in xrange(n_fre):
                for c in xrange(n_cor0):
                    if sum_ww[row, f, c]:
                        vis[row, f, c] /= sum_ww[row, f, c]


@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def rebin_model(fcomplex [:,:,:] model,  const fcomplex [:,:,:] model0,
                  const flag_t [:,:,:] flag0,
                  const index_t [:] rebin_row_map, const index_t [:] rebin_chan_map):
    """
    Rebins a model column, following a rebin_row_map precomputed by cyrebin_vis
    """

    cdef int i, t, f, f0, c
    cdef int n_row0, n_row, n_fre0, n_cor0, n_fre, n_cor, ts_in, ts_out, a1, a2, row0, row
    cdef int conjugate

    n_row0, n_fre0, n_cor0 = model0.shape[0], model0.shape[1], model0.shape[2]
    n_row, n_fre, n_cor = model.shape[0], model0.shape[1], n_cor0

    counts0 = np.zeros((n_row,n_fre,n_cor), np.int64)    # counts per each model visibility
    cdef np.int64_t [:,:,:] counts = counts0

    for row0 in xrange(n_row0):
        row = rebin_row_map[row0]
        conjugate = row<0
        if conjugate:
            row = -row

        for f0 in xrange(n_fre0):
            f = rebin_chan_map[f0]
            for c in xrange(n_cor0):
                if not flag0[row0, f0, c]:
                    counts[row, f, c] += 1
                    model[row, f, c] += model0[row0, f0, c].conjugate() if conjugate else model0[row0, f0, c]

    # now normalize by counts
    for row in xrange(n_row):
        for f in xrange(n_fre):
            for c in xrange(n_cor):
                if counts[row, f, c]:
                    model[row, f, c] /= counts[row, f, c]

