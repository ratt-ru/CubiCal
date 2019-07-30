# CubiCal: a radio interferometric calibration suite
# (c) 2017 Rhodes University & Jonathan S. Kenyon
# http://github.com/ratt-ru/CubiCal
# This code is distributed under the terms of GPLv2, see LICENSE.md for details
"""
Cython kernels for Mad Max flagger
"""

import numpy as np
import numpy.ma
cimport numpy as np
import cython
from cython.parallel import parallel, prange, threadid
import cubical.kernels

ctypedef np.complex64_t fcomplex
ctypedef np.complex128_t dcomplex

ctypedef fused float3264:
    np.float32_t
    np.float64_t

ctypedef fused complex3264:
    fcomplex
    dcomplex

ctypedef np.uint16_t flag_t

cygenerics = cubical.kernels.import_kernel("cygenerics")



@cython.boundscheck(False)
@cython.nonecheck(False)
cdef inline void ELEM_SWAP(float3264 * arr, int i, int j) nogil:
    cdef float3264 tmp
    tmp = arr[i]
    arr[i] = arr[j]
    arr[j] = tmp

@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
cdef float3264 quick_select(float3264 * arr, int n) nogil:
    cdef int low, high
    cdef int median
    cdef int middle, ll, hh

    low = 0
    high = n-1
    median = (low + high) / 2

    while True:
        if high <= low:             # one element
            return arr[median]

        if high == low + 1:         # two elements only
            if arr[low] > arr[high]:
                ELEM_SWAP(arr, low, high)
            return arr[median]

        # Find median of low, middle and high items; swap into position low
        middle = (low + high) / 2;
        if arr[middle] > arr[high]:
            ELEM_SWAP(arr, middle, high)
        if arr[low] > arr[high]:
            ELEM_SWAP(arr, low, high)
        if arr[middle] > arr[low]:
            ELEM_SWAP(arr, middle, low)

        # Swap low item (now in position middle) into position (low+1)
        ELEM_SWAP(arr, middle, low+1)

        # Nibble from each end towards middle, swapping items when stuck
        ll = low + 1
        hh = high
        while True:
            ll += 1
            while arr[low] > arr[ll]:
                ll += 1
            hh -= 1
            while arr[hh]  > arr[low]:
                hh -= 1

            if hh < ll:
                break

            ELEM_SWAP(arr, ll, hh)

        # Swap middle item (in position low) back into correct position
        ELEM_SWAP(arr, low, hh)

        # Re-set active partition
        if hh <= median:
            low = ll
        if hh >= median:
            high = hh - 1


def get_corr_arr(diag, offdiag):
    # work out which correlations to loop over
    if diag:
        if offdiag:
            corr_list = [ [c1, c2] for c1 in xrange(2) for c2 in xrange(2) ]
        else:
            corr_list = [ [0, 0], [1, 1] ]
    elif offdiag:
            corr_list = [ [0, 1], [1, 0] ]
    else:
        raise ValueError("diag and/or offdiag must be set")
    return np.array(corr_list, np.int32)


@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def compute_mad(float3264 [:,:,:,:,:,:,:] absres, flag_t [:,:,:,:] flags,int diag=1,int offdiag=1):
    cdef int n_mod, n_tim, n_fre, n_ant, bl, aa, ab, m, ic, c1, c2, t, f, thread, nval
    cdef np.float32_t x

    n_mod = absres.shape[0]
    n_tim = absres.shape[1]
    n_fre = absres.shape[2]
    n_ant = absres.shape[3]

    cdef np.int32_t [:,:] baselines = cygenerics.half_baselines(n_ant)
    cdef int n_bl = baselines.shape[0]
    cdef int num_threads = cubical.kernels.num_omp_threads

    # work out which correlations to loop over
    corr_arr = get_corr_arr(diag, offdiag)
    cdef np.int32_t [:,:] corr = corr_arr
    cdef int n_cor = corr.shape[0]

    absvals_arr = np.empty((num_threads or 1, n_cor*n_tim*n_fre), np.float32)
    mad_arr = np.zeros((n_mod, n_ant, n_ant), np.float32)
    mad_arr_fl = np.ones_like(mad_arr, np.uint8)

    valid_arr = np.ones((n_mod, n_tim, n_fre, n_ant, n_ant), np.uint8)
    cdef np.uint8_t   [:,:,:,:,:]  valid = valid_arr

    cdef np.float32_t [:,:] absvals = absvals_arr
    cdef np.float32_t [:,:,:] mad = mad_arr
    cdef np.uint8_t  [:,:,:] madfl = mad_arr_fl

    with nogil, parallel(num_threads=num_threads):
        for bl in prange(n_bl, schedule='static'):
            thread = threadid()
            aa = baselines[bl][0]
            ab = baselines[bl][1]
            for m in xrange(n_mod):
                nval = 0
                for ic in xrange(n_cor):
                    c1 = corr[ic][0]
                    c2 = corr[ic][1]
                    # get list of non-flagged absolute values
                    for t in xrange(n_tim):
                        for f in xrange(n_fre):
                            x = absres[m, t, f, aa, ab, c1, c2]
                            if x != 0 and not flags[t, f, aa, ab]:
                                absvals[thread, nval] = x
                                nval = nval+1
                            else:
                                valid[m,t,f,aa,ab] = valid[m,t,f,ab,aa] = 0
                # do quick-select
                if nval:
                    mad[m, aa, ab] = mad[m, ab, aa] = quick_select(&(absvals[thread, 0]), nval)
                    madfl[m, aa, ab] = madfl[m, ab, aa] = False

    return np.ma.masked_array(mad_arr, mad_arr_fl, fill_value=0), valid_arr



@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def compute_mad_per_corr(float3264 [:,:,:,:,:,:,:] absres, flag_t [:,:,:,:] flags,int diag=1,int offdiag=1):
    cdef int n_mod, n_tim, n_fre, n_ant, bl, aa, ab, m, ic, c1, c2, t, f, thread, nval
    cdef np.float32_t x

    n_mod = absres.shape[0]
    n_tim = absres.shape[1]
    n_fre = absres.shape[2]
    n_ant = absres.shape[3]

    cdef np.int32_t [:,:] baselines = cygenerics.half_baselines(n_ant)
    cdef int n_bl = baselines.shape[0]
    cdef int num_threads = cubical.kernels.num_omp_threads

    # work out which correlations to loop over
    corr_arr = get_corr_arr(diag, offdiag)
    cdef np.int32_t [:,:] corr = corr_arr
    cdef int n_cor = corr.shape[0]

    absvals_arr = np.empty((num_threads or 1, n_tim*n_fre), np.float32)
    mad_arr = np.zeros((n_mod, n_ant, n_ant, 2, 2), np.float32)
    mad_arr_fl = np.ones_like(mad_arr, np.uint8)

    valid_arr = np.ones((n_mod, n_tim, n_fre, n_ant, n_ant), np.uint8)
    cdef np.uint8_t   [:,:,:,:,:]  valid = valid_arr

    cdef np.float32_t [:,:] absvals = absvals_arr
    cdef np.float32_t [:,:,:,:,:] mad = mad_arr
    cdef np.uint8_t   [:,:,:,:,:] madfl = mad_arr_fl

    with nogil, parallel(num_threads=num_threads):
        for bl in prange(n_bl, schedule='static'):
            thread = threadid()
            aa = baselines[bl][0]
            ab = baselines[bl][1]
            for m in xrange(n_mod):
                for ic in xrange(n_cor):
                    c1 = corr[ic][0]
                    c2 = corr[ic][1]
                    # get list of non-flagged absolute values
                    nval=0
                    for t in xrange(n_tim):
                        for f in xrange(n_fre):
                            x = absres[m, t, f, aa, ab, c1, c2]
                            if x != 0 and not flags[t, f, aa, ab]:
                                absvals[thread, nval] = x
                                nval = nval+1
                            else:
                                valid[m,t,f,aa,ab] = valid[m,t,f,ab,aa] = 0
                    # do quick-select
                    if nval:
                        mad[m, aa, ab, c1, c2] = mad[m, ab, aa, c2, c1] = quick_select(&(absvals[thread, 0]), nval)
                        madfl[m, aa, ab, c1, c2] = madfl[m, ab, aa, c1, c2] = False

    return np.ma.masked_array(mad_arr, mad_arr_fl, fill_value=0), valid_arr


@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def threshold_mad (float3264 [:,:,:,:,:,:,:] absres, np.float32_t [:,:,:,:,:] thr, flag_t [:,:,:,:] flags, flag_t flagbit, np.uint8_t [:,:,:,:,:] goodies,int diag=1,int offdiag=1):
    cdef int n_mod, n_tim, n_fre, n_ant, bl, aa, ab, m, ic, c1, c2, t, f, thread, nval
    cdef np.float32_t x

    n_mod = absres.shape[0]
    n_tim = absres.shape[1]
    n_fre = absres.shape[2]
    n_ant = absres.shape[3]

    cdef np.int32_t [:,:] baselines = cygenerics.half_baselines(n_ant)
    cdef int n_bl = baselines.shape[0]
    cdef int num_threads = cubical.kernels.num_omp_threads

    baddies_arr = np.zeros((n_tim,n_fre,n_ant,n_ant), np.uint8)
    cdef np.uint8_t [:,:,:,:] baddies = baddies_arr
    cdef int good_models, flagged

    # work out which correlations to loop over
    corr_arr = get_corr_arr(diag, offdiag)
    cdef np.int32_t [:,:] corr = corr_arr
    cdef int n_cor = corr.shape[0]

    with nogil, parallel(num_threads=num_threads):
        for bl in prange(n_bl, schedule='static'):
            thread = threadid()
            aa = baselines[bl][0]
            ab = baselines[bl][1]
            for t in xrange(n_tim):
                for f in xrange(n_fre):
                    good_models = flagged = 0
                    for m in xrange(n_mod):
                        if goodies[m,t,f,aa,ab]:
                            good_models = 1
                            # if any correlation is over threshold, break out (will flag all four)
                            for ic in xrange(n_cor):
                                c1 = corr[ic][0]
                                c2 = corr[ic][1]
                                if absres[m,t,f,aa,ab,c1,c2] > thr[m,aa,ab,c1,c2]:
                                    flagged = 1
                                    break
                    # raise flag only if we had a valid residual in at least one model to begin with
                    if flagged and good_models:
                        baddies[t,f,aa,ab] = baddies[t,f,ab,aa] = 1
                        flags[t,f,aa,ab] = flags[t,f,ab,aa] = flags[t,f,ab,aa] | flagbit
                        for m in xrange(n_mod):
                            goodies[m,t,f,aa,ab] = goodies[m,t,f,ab,aa] = 0

    return baddies_arr

