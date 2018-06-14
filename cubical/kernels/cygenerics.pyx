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

ctypedef np.uint16_t flag_t


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

cdef extern from *:
    """
    /*
     *  This Quickselect routine is based on the algorithm described in
     *  "Numerical recipes in C", Second Edition,
     *  Cambridge University Press, 1992, Section 8.5, ISBN 0-521-43108-5
     *  This code by Nicolas Devillard - 1998. Public domain.
     */
    typedef float elem_type;

    #define ELEM_SWAP(a,b) { register elem_type t=(a);(a)=(b);(b)=t; }

    elem_type quick_select(elem_type arr[], int n)
    {
        int low, high ;
        int median;
        int middle, ll, hh;

        low = 0 ; high = n-1 ; median = (low + high) / 2;
        for (;;) {
            if (high <= low) /* One element only */
                return arr[median] ;

            if (high == low + 1) {  /* Two elements only */
                if (arr[low] > arr[high])
                    ELEM_SWAP(arr[low], arr[high]) ;
                return arr[median] ;
            }

        /* Find median of low, middle and high items; swap into position low */
        middle = (low + high) / 2;
        if (arr[middle] > arr[high])    ELEM_SWAP(arr[middle], arr[high]) ;
        if (arr[low] > arr[high])       ELEM_SWAP(arr[low], arr[high]) ;
        if (arr[middle] > arr[low])     ELEM_SWAP(arr[middle], arr[low]) ;

        /* Swap low item (now in position middle) into position (low+1) */
        ELEM_SWAP(arr[middle], arr[low+1]) ;

        /* Nibble from each end towards middle, swapping items when stuck */
        ll = low + 1;
        hh = high;
        for (;;) {
            do ll++; while (arr[low] > arr[ll]) ;
            do hh--; while (arr[hh]  > arr[low]) ;

            if (hh < ll)
            break;

            ELEM_SWAP(arr[ll], arr[hh]) ;
        }

        /* Swap middle item (in position low) back into correct position */
        ELEM_SWAP(arr[low], arr[hh]) ;

        /* Re-set active partition */
        if (hh <= median)
            low = ll;
            if (hh >= median)
            high = hh - 1;
        }
    }

    #undef ELEM_SWAP
    """

#
#@cython.cdivision(True)
#@cython.wraparound(False)
#@cython.boundscheck(False)
#@cython.nonecheck(False)
#def cycompute_mad(fcomplex [:,:,:,:,:,:,:] resid, flag_t [:,:,:,:] flags):
#    cdef int n_mod, n_tim, n_fre, n_ant, n_cor, bl, aa, ab, m, c1, c2, t, f, nval
#
#    n_mod, n_tim, n_fre, n_ant, n_ant, n_cor, n_cor = resid.shape
#
#    absvals_arr = np.empty((n_mod, n_ant, n_ant, n_cor, n_cor, n_tim*n_fre), resid.real.dtype)
#    mad_arr = np.zeros((n_mod, n_ant, n_ant, n_cor, n_cor), resid.real.dtype)
#
#    cdef fcomplex [:,:,:,:,:,:] absvals = absvals_arr
#    cdef np.float32_t [:,:,:,:,:] mad = mad_arr
#
#
#    cdef np.int32_t [:,:] baselines = half_baselines(n_ant)
#    cdef n_bl = half_baselines.shape[0]
#    assert(flags.dtype == np.complex64)
#    cdef int num_threads = cubical.kernels.num_omp_threads
#
#    with nogil, parallel(num_threads=num_threads):
#        for bl in prange(n_bl, schedule='static'):
#            aa, ab = half_baselines[bl]
#            for m in xrange(n_mod):
#                for c1 in xrange(2):
#                    for c2 in xrange(2):
#                        # get list of non-flagged absolute values
#                        nval=0
#                        for t in xrange(n_tim):
#                            for f in xrange(n_freq):
#                                if not flags[t, f, aa, ab]:
#                                    absvals[m, aa, ab, c1, c2, nval] = abs(resid[m,t,f,aa,ab,c1,c2])
#                                    nval = nval+1
#                        # do quick-select
#                        if nval:
#                            mad[m, aa, ab, c1, c2] = quick_select(&(absvals[m, aa, ab, c1, c2]), nval)
#
#    return mad_arr
