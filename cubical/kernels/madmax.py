# CubiCal: a radio interferometric calibration suite
# (c) 2017 Rhodes University & Jonathan S. Kenyon
# http://github.com/ratt-ru/CubiCal
# This code is distributed under the terms of GPLv2, see LICENSE.md for details
"""
Cython kernels for Mad Max flagger
"""
from builtins import range

import numpy as np
from numba import jit, prange
import cubical.kernels

use_parallel = True if cubical.kernels.num_omp_threads > 1 else False
use_cache = cubical.kernels.use_cache

# Retain the following in case in becomes practical to optimise this ourselves.

# @jit(nopython=True, fastmath=True, parallel=False, cache=False, nogil=True)
# def quick_select(arr, n):

#     low = 0
#     high = n-1
#     median = (low + high) // 2

#     while True:
#         if high <= low:             # one element
#             return arr[median]

#         if high == low + 1:         # two elements only
#             if arr[low] > arr[high]:
#                 arr[low], arr[high] = arr[high], arr[low]
#             return arr[median]

#         # Find median of low, middle and high items; swap into position low
#         middle = (low + high) // 2;
#         if arr[middle] > arr[high]:
#             arr[middle], arr[high] = arr[high], arr[middle]
#         if arr[low] > arr[high]:
#             arr[low], arr[high] = arr[high], arr[low]
#         if arr[middle] > arr[low]:
#             arr[middle], arr[low] = arr[low], arr[middle]

#         # Swap low item (now in position middle) into position (low+1)
#         arr[middle], arr[low+1] = arr[low+1], arr[middle]

#         # Nibble from each end towards middle, swapping items when stuck
#         ll = low + 1
#         hh = high
#         while True:
#             ll += 1
#             while arr[low] > arr[ll]:
#                 ll += 1
#             hh -= 1
#             while arr[hh]  > arr[low]:
#                 hh -= 1

#             if hh < ll:
#                 break

#             arr[ll], arr[hh] = arr[hh], arr[ll]

#         # Swap middle item (in position low) back into correct position
#         arr[hh], arr[low] = arr[low], arr[hh]

#         # Re-set active partition
#         if hh <= median:
#             low = ll
#         if hh >= median:
#             high = hh - 1

def compute_mad(absres, flags, diag=1, offdiag=1):
    """Wrapper function for non-numba functionality."""

    mad_arr, mad_arr_fl, valid_arr = compute_mad_internals(absres, flags, diag, offdiag)

    return np.ma.masked_array(mad_arr, mad_arr_fl, fill_value=0), valid_arr

@jit(nopython=True, fastmath=True, parallel=use_parallel, cache=use_cache, nogil=True)
def compute_mad_internals(absres, flags, diag=1, offdiag=1):
    """
    Given the absolute values of the residuals and a flag array, computes the per-antenna mad estimates. 
    For the per-antenna, per-correlation variant, see compute_mad_per_corr. Keyword args establish which 
    correlations to include in the median computation.

    Args:
        absres (np.float32 or np.float64):
            Absolute value residual array with dimensions (m, t, f, a, a, c, c).
        flags (np.uint16):
            Flag array with dimension (t, f, a, a).
        diag (int/bool):
            Include diagonal correlations.
        offdiag (int):
            Include off-diagonal correlations.
    """

    n_mod = absres.shape[0]
    n_tim = absres.shape[1]
    n_fre = absres.shape[2]
    n_ant = absres.shape[3]

    bls = np.array([[i,j] for i in range(n_ant) for j in range(i+1, n_ant)], dtype=np.int32)
    n_bl = bls.shape[0]

    # Check for valid correlation parameters.
    if diag==0 and offdiag==0:
        raise ValueError("Diag and/or offdiag must be set.")

    # Build correlations to loop over.
    if diag and offdiag:
        corr_list = [[0,0], [0,1], [1,0], [1,1]]
    elif diag:
        corr_list = [[0,0], [1,1]]
    else:
        corr_list = [[0,1], [1,0]]

    corr = np.array(corr_list, dtype=np.int32)
    n_cor = corr.shape[0]

    mad_shape = (n_mod, n_ant, n_ant)
    mad_arr = np.zeros(mad_shape, np.float32)
    mad_arr_fl = np.full(mad_shape, True)

    valid_arr = np.ones((n_mod, n_tim, n_fre, n_ant, n_ant), np.uint8)

    for ibl in prange(n_bl):
        aa, ab = bls[ibl][0], bls[ibl][1]  
        for m in range(n_mod):
            valid_vals = np.empty((n_cor*n_tim*n_fre), np.float32)
            n_valid_vals = 0
            for ic in range(n_cor):
                c1, c2 = corr[ic][0], corr[ic][1] # Correlation indices.
                for t in range(n_tim):
                    for f in range(n_fre):
                        # Select a single correlation from absres. 
                        val = absres[m, t, f, aa, ab, c1, c2]
                        # If it is non-zero and unflagged, add it to the array.
                        # Otherwise, mark as invalid.
                        if val != 0 and not flags[t, f, aa, ab]:
                            valid_vals[n_valid_vals] = val
                            n_valid_vals += 1
                        else:
                            valid_arr[m,t,f,aa,ab] = valid_arr[m,t,f,ab,aa] = 0

            # Find the median value of baseline (aa,ab) over valid times, frequencies and corellations.
            if n_valid_vals:
                mad_arr[m, aa, ab] = mad_arr[m, ab, aa] = np.median(valid_vals[:n_valid_vals])
                mad_arr_fl[m, aa, ab] = mad_arr_fl[m, ab, aa] = False

    return mad_arr, mad_arr_fl, valid_arr

def compute_mad_per_corr(absres, flags, diag=1, offdiag=1):
    """Wrapper function for non-numba functionality."""
    
    mad_arr, mad_arr_fl, valid_arr = compute_mad_per_corr_internals(absres, flags, diag, offdiag)

    return np.ma.masked_array(mad_arr, mad_arr_fl, fill_value=0), valid_arr

@jit(nopython=True, fastmath=True, parallel=use_parallel, cache=use_cache, nogil=True)
def compute_mad_per_corr_internals(absres, flags, diag=1, offdiag=1):
    """
    Given the absolute values of the residuals and a flag array, computes the per-antenna, per-correlation 
    mad estimates. Keyword args establish which correlations to include in the median computation.

    Args:
        absres (np.float32 or np.float64):
            Absolute value residual array with dimensions (m, t, f, a, a, c, c).
        flags (np.uint16):
            Flag array with dimension (t, f, a, a).
        diag (int/bool):
            Include diagonal correlations.
        offdiag (int):
            Include off-diagonal correlations.
    """

    n_mod = absres.shape[0]
    n_tim = absres.shape[1]
    n_fre = absres.shape[2]
    n_ant = absres.shape[3]

    bls = np.array([[i,j] for i in range(n_ant) for j in range(i+1, n_ant)], dtype=np.int32)
    n_bl = bls.shape[0]

    # Check for valid correlation parameters.
    if diag==0 and offdiag==0:
        raise ValueError("Diag and/or offdiag must be set.")

    # Build correlations to loop over.
    if diag and offdiag:
        corr_list = [[0,0], [0,1], [1,0], [1,1]]
    elif diag:
        corr_list = [[0,0], [1,1]]
    else:
        corr_list = [[0,1], [1,0]]

    corr = np.array(corr_list, dtype=np.int32)
    n_cor = corr.shape[0]

    mad_shape = (n_mod, n_ant, n_ant, 2, 2)
    mad_arr = np.zeros(mad_shape, np.float32)
    mad_arr_fl = np.full(mad_shape, True)

    valid_arr = np.ones((n_mod, n_tim, n_fre, n_ant, n_ant), np.uint8)

    for ibl in prange(n_bl):
        aa, ab = bls[ibl][0], bls[ibl][1]  
        for m in range(n_mod):
            valid_vals = np.empty((n_cor*n_tim*n_fre), np.float32)
            for ic in range(n_cor):
                c1, c2 = corr[ic][0], corr[ic][1] # Correlation indices.
                n_valid_vals = 0
                for t in range(n_tim):
                    for f in range(n_fre):
                        # Select a single correlation from absres. 
                        val = absres[m, t, f, aa, ab, c1, c2]
                        # If it is non-zero and unflagged, add it to the array.
                        # Otherwise, mark as invalid.
                        if val != 0 and not flags[t, f, aa, ab]:
                            valid_vals[n_valid_vals] = val
                            n_valid_vals += 1
                        else:
                            valid_arr[m,t,f,aa,ab] = valid_arr[m,t,f,ab,aa] = 0

                # Find the median value of baseline (aa,ab) over valid times, frequencies and corellations.
                if n_valid_vals:
                    mad_arr[m, aa, ab, c1, c2] = mad_arr[m, ab, aa, c2, c1] = np.median(valid_vals[:n_valid_vals])
                    mad_arr_fl[m, aa, ab, c1, c2] = mad_arr_fl[m, ab, aa, c1, c2] = False

    return mad_arr, mad_arr_fl, valid_arr

@jit(nopython=True, fastmath=True, parallel=use_parallel, cache=use_cache, nogil=True)
def threshold_mad(absres, thr, flags, flagbit, valid_arr, diag=1, offdiag=1):
    """
    Given the absolute values of the residuals and an array of thresholds, thresholds the residuals.
    Keyword args establish which correlations to include in the median computation.

    Args:
        absres (np.float32 or np.float64):
            Absolute value residual array with dimensions (m, t, f, a, a, c, c).
        thr (np.float32 or np.float64):
            Threshold array with dimensions (m, a, a, c, c).
        flags (np.uint16):
            Flag array with dimension (t, f, a, a).
        flagbit (int):
            Bitflag to raise.
        valid_arr (np.uint16):
            Flag array for valid_arr absres entries with dimension (m, t, f, a, a).
        diag (int/bool):
            Include diagonal correlations.
        offdiag (int):
            Include off-diagonal correlations.
    """

    n_mod = absres.shape[0]
    n_tim = absres.shape[1]
    n_fre = absres.shape[2]
    n_ant = absres.shape[3]

    bls = np.array([[i,j] for i in range(n_ant) for j in range(i+1, n_ant)], dtype=np.int32)
    n_bl = bls.shape[0]

    invalid_arr = np.zeros((n_tim, n_fre, n_ant, n_ant), np.uint8)

    # Check for valid correlation parameters.
    if diag==0 and offdiag==0:
        raise ValueError("Diag and/or offdiag must be set.")

    # Build correlations to loop over.
    if diag and offdiag:
        corr_list = [[0,0], [0,1], [1,0], [1,1]]
    elif diag:
        corr_list = [[0,0], [1,1]]
    else:
        corr_list = [[0,1], [1,0]]

    corr = np.array(corr_list, dtype=np.int32)
    n_cor = corr.shape[0]

    for ibl in prange(n_bl):
        aa, ab = bls[ibl][0], bls[ibl][1]  
        for t in range(n_tim):
            for f in range(n_fre):
                good_models = flagged = 0
                for m in range(n_mod):
                    if valid_arr[m,t,f,aa,ab]:
                        good_models = 1
                        # If any correlation exceeds threshold, break out (flagging all four).
                        for ic in range(n_cor):
                            c1, c2 = corr[ic][0], corr[ic][1]
                            if absres[m,t,f,aa,ab,c1,c2] > thr[m,aa,ab,c1,c2]:
                                flagged = 1
                                break
                # Raise flags only if we had a valid residual in at least one model to begin with.
                if flagged and good_models:
                    invalid_arr[t,f,aa,ab] = invalid_arr[t,f,ab,aa] = 1
                    flags[t,f,aa,ab] = flags[t,f,ab,aa] = flags[t,f,ab,aa] | flagbit
                    for m in range(n_mod):
                        valid_arr[m,t,f,aa,ab] = valid_arr[m,t,f,ab,aa] = 0

    return invalid_arr

