# CubiCal: a radio interferometric calibration suite
# (c) 2017 Rhodes University & Jonathan S. Kenyon
# http://github.com/ratt-ru/CubiCal
# This code is distributed under the terms of GPLv2, see LICENSE.md for details
"""
Cython kernels for various rebinning operations. Common dimensions of arrays are:

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
from builtins import range

import numpy as np
from numba import jit, prange

import cubical.kernels

use_parallel = True if cubical.kernels.num_omp_threads > 1 else False
use_cache = cubical.kernels.use_cache

@jit(nopython=True, fastmath=True, parallel=use_parallel, cache=use_cache, nogil=True)
def rebin_index_columns(time, time0, antea, antea0, anteb, anteb0,
             			ddid_col, ddid_col0, row_map):
    """
    Rebins indexing columns (time, ant1/2, ddid), using the given row map.
    Time should be a zero-filled output column of the expected length (row_map[-1]+1)
    anteab, anteb, ddid_col should be empty output columns of the expected length.
    """

    n_input_rows = time0.shape[0]
    n_output_rows  = time.shape[0]

    vis_count = np.zeros(n_output_rows, np.int64)

    for input_row in range(n_input_rows):
        output_row = abs(row_map[input_row])
        vis_count[output_row] += 1
        time[output_row] += time0[input_row]
        antea[output_row] = antea0[input_row]
        anteb[output_row] = anteb0[input_row]
        ddid_col[output_row] = ddid_col0[input_row]

    for output_row in range(n_output_rows):
        if vis_count[output_row]:
            time[output_row] /= vis_count[output_row]

@jit(nopython=True, fastmath=True, parallel=use_parallel, cache=use_cache, nogil=True)
def rebin_vis(vis, vis0, uvw, uvw0, flag, flag0, weights, weights0, num_weights,
              rebin_row_map, rebin_chan_map):
    """
    Rebin the input data.
    """
    n_input_rows, n_fre0, n_cor0 = vis0.shape[0], vis0.shape[1], vis0.shape[2]
    n_row, n_fre, n_cor = vis.shape[0], vis.shape[1], n_cor0

    sum_ww = np.zeros((n_row,n_fre,n_cor), np.float32)    # Sum of weights per each output visibility
    sum_rw = np.zeros(n_row, np.float32)                  # Sum of weights per each output row

    for row0 in range(n_input_rows):
        row = rebin_row_map[row0]
        conjugate = row<0
        if conjugate:
            row = -row

        row_sum_weights = 0  # sum of weights of current _input_ row

        for f0 in range(n_fre0):
            f = rebin_chan_map[f0]
            for c in range(n_cor0):
		# Output flags accumulate all input flags across the bin with bitwise-OR.
		# However, below we'll clear them if at least one unflagged visibility was present.
                flag[row, f, c] |= flag0[row0, f0, c]  
                if not flag0[row0, f0, c]:
                    # accumulate weights
                    if num_weights:
                        for w in range(num_weights):
                            weights[w, row, f, c] += weights0[w, row0, f0, c]
                        ww = weights0[0, row0, f0, c]
                    else:
                        ww = 1
                    vis[row, f, c] += ww*vis0[row0, f0, c].conjugate() if conjugate else ww*vis0[row0, f0, c]
                    sum_ww[row, f, c] += ww
                    sum_rw[row] += ww
                    row_sum_weights += ww
        if row_sum_weights:
            for i in range(3):
                uvw[row, i] += row_sum_weights*uvw0[row0, i]

    # Now normalize by counts and sums of the weights.
    for row in range(n_row):
        if sum_rw[row]:
            for i in range(3):
                uvw[row, i] /= sum_rw[row]
            for f in range(n_fre):
                for c in range(n_cor0):
                    if sum_ww[row, f, c]:
                        flag[row, f, c] = 0
                        vis[row, f, c] /= sum_ww[row, f, c]

@jit(nopython=True, fastmath=True, parallel=use_parallel, cache=use_cache, nogil=True)
def rebin_model(model, model0, flag0, rebin_row_map, rebin_chan_map):
    """
    Rebins a model column, following a rebin_row_map precomputed by cyrebin_vis
    """
    n_row0, n_fre0, n_cor0 = model0.shape[0], model0.shape[1], model0.shape[2]
    n_row, n_fre, n_cor = model.shape[0], model0.shape[1], n_cor0

    counts = np.zeros((n_row,n_fre,n_cor), np.int64)    # counts per each model visibility

    for row0 in range(n_row0):
        row = rebin_row_map[row0]
        conjugate = row<0
        if conjugate:
            row = -row

        for f0 in range(n_fre0):
            f = rebin_chan_map[f0]
            for c in range(n_cor0):
                if not flag0[row0, f0, c]:
                    counts[row, f, c] += 1
                    model[row, f, c] += model0[row0, f0, c].conjugate() if conjugate else model0[row0, f0, c]

    # now normalize by counts
    for row in range(n_row):
        for f in range(n_fre):
            for c in range(n_cor):
                if counts[row, f, c]:
                    model[row, f, c] /= counts[row, f, c]


