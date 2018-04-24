# CubiCal: a radio interferometric calibration suite
# (c) 2017 Rhodes University & Jonathan S. Kenyon
# http://github.com/ratt-ru/CubiCal
# This code is distributed under the terms of GPLv2, see LICENSE.md for details
"""
Cython kernels for the full-complex 2x2 gain machine. Functions require output arrays to be 
provided. Common dimensions of arrays are:

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


ctypedef fused complex3264:
    np.complex64_t
    np.complex128_t

def cyallocate_DTFACC(shape, dtype):
    """
    Allocates an array of shape NDxNTxNFxNAxNC, with its underlying memory layout optimized to the kernel
    """
    nd,nt,nf,na,nc,_ = shape
    _intrinsic_shape = [na,nd,nt,nf,nc,nc]
    return np.empty(_intrinsic_shape, dtype=dtype).transpose((1,2,3,0,4,5))


cdef inline void gm_product(complex3264 * out,const complex3264 *g,const complex3264 *m) nogil:
    """
    Computes a 2x2 matrix product: out = G.M. G is diagonal.
    A matrix is just a sequence in memory of four complex numbers [x00,x01,x10,x11] (so e.g. the last 2,2 axes of a cube)

    A note on parameter names for these matrix functions: we'll use A,B,C for generic matrices, and
    G (gains) and M (model) and J (Jacobian element) when functions are only used with matrices that
    have a specific role. This is to simplify the generation of kernels optimized down to diagonal G and
    diagonal M cases.

    We have two simplified cases: G diagonal but visibilities 2x2, or both diagonal. In the former case
    only some of these functions get simplified, in the latter case, all do.
    """
    out[0] = g[0]*m[0]
    out[1] = g[0]*m[1]
    out[2] = g[3]*m[2]
    out[3] = g[3]*m[3]

cdef inline void mat_product(complex3264 * out,const complex3264 *a,const complex3264 *b) nogil:
    """
    Computes a 2x2 matrix product: out =  A.B
    This is used for the (J^HR)(J^HJ)^{-1} product. This looks same as gm_product()
    """
    out[0] = (a[0]*b[0] + a[1]*b[2])
    out[1] = (a[0]*b[1] + a[1]*b[3])
    out[2] = (a[2]*b[0] + a[3]*b[2])
    out[3] = (a[2]*b[1] + a[3]*b[3])

cdef inline void add_rjh_product(complex3264 * out,const complex3264 *r,const complex3264 *jh) nogil:
    """
    Adds a 2x2 matrix product in place: out += R.J
    """
    out[0] += (r[0]*jh[0] + r[1]*jh[2])
    out[1] += (r[0]*jh[1] + r[1]*jh[3])
    out[2] += (r[2]*jh[0] + r[3]*jh[2])
    out[3] += (r[2]*jh[1] + r[3]*jh[3])

cdef inline void add_jhj_product(complex3264 * out,const complex3264 *j) nogil:
    """
    Adds a matrix conjugate product in place: out += J^H.J
    """
    out[0] += (j[0].conjugate()*j[0] + j[2].conjugate()*j[2])
    out[1] += (j[0].conjugate()*j[1] + j[2].conjugate()*j[3])
    out[2] += (j[1].conjugate()*j[0] + j[3].conjugate()*j[2])
    out[3] += (j[1].conjugate()*j[1] + j[3].conjugate()*j[3])

cdef inline void gmgh_product(complex3264 * out,const complex3264 *g,const complex3264 *m,const complex3264 *gh) nogil:
    """
    Computes a triple 2x2 matrix product: out = G.M.G^H
    """
    out[0] = g[0]*m[0]*gh[0]
    out[1] = g[0]*m[1]*gh[3]
    out[2] = g[3]*m[2]*gh[0]
    out[3] = g[3]*m[3]*gh[3]

cdef inline void subtract_gmgh_product(complex3264 * out,const complex3264 *g,const complex3264 *m,const complex3264 *gh) nogil:
    """
    Subtracts a triple 2x2 matrix product: out -= G.M.G^H
    """
    out[0] -= g[0]*m[0]*gh[0]
    out[1] -= g[0]*m[1]*gh[3]
    out[2] -= g[3]*m[2]*gh[0]
    out[3] -= g[3]*m[3]*gh[3]

cdef inline void inplace_gmgh_product(const complex3264 *g,complex3264 *m,const complex3264 *gh) nogil:
    """
    Computes a triple 2x2 matrix product in place: M = G.M.G^H
    """
    gmgh_product(g,m,gh)

cdef inline void mat_conjugate(complex3264 * out,const complex3264 *x) nogil:
    """
    Computes a 2x2 matrix Hermitian conjugate: out = X^H
    """
    out[0] = x[0].conjugate()
    out[1] = x[2].conjugate()
    out[2] = x[1].conjugate()
    out[3] = x[3].conjugate()


cdef int half_baselines[16384][2]                           # MAX NUMBER OF BASELINES HERE
cdef inline int compute_half_baselines(int n_ant):
    """
    Computes array of unique baseline indices, given a number of antennas
    """
    i = 0
    for p in xrange(n_ant-1):
        for q in xrange(p+1, n_ant):
            half_baselines[i][0] = p
            half_baselines[i][1] = q
            i += 1
    return n_ant*(n_ant-1)/2

cdef int all_baselines[32768][2]                           # MAX NUMBER OF BASELINES HERE
cdef inline int compute_all_baselines(int n_ant):
    """
    Computes array of unique baseline indices, given a number of antennas
    """
    i = 0
    for p in xrange(n_ant):
        for q in xrange(n_ant):
            all_baselines[i][0] = p
            all_baselines[i][1] = q
            i += 1
    return n_ant*n_ant


@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def cycompute_residual(complex3264 [:,:,:,:,:,:,:,:] m,
                       complex3264 [:,:,:,:,:,:] g,
                       complex3264 [:,:,:,:,:,:] gh,
                       complex3264 [:,:,:,:,:,:,:] r,
                       int t_int,
                       int f_int):

    """
    Given the model, gains, and their conjugates, computes the residual. Residual has full time and
    frequency resolution - solution intervals are used to correctly associate the gains with the
    model.

    Args:
        m (np.complex64 or np.complex128):
            Typed memoryview of model array with dimensions (d, m, t, f, a, a, c, c).
        g (np.complex64 or np.complex128):
            Typed memoryview of gain array with dimension (d, ti, fi, a, c, c).
        gh (np.complex64 or np.complex128):
            Typed memoryview of conjugate gain array with dimensions (d, ti, fi, a, c, c).
        r (np.complex64 or np.complex128):
            Typed memoryview of residual array with dimensions (m, t, f, a, a, c, c).
        t_int (int):
            Number of time slots per solution interval.
        f_int (int):
            Number of frequencies per solution interval.
    """

    cdef int d, i, t, f, aa, ab, ibl, rr, rc = 0
    cdef int n_dir, n_mod, n_tim, n_fre, n_ant

    n_dir = m.shape[0]
    n_mod = m.shape[1]
    n_tim = m.shape[2]
    n_fre = m.shape[3]
    n_ant = m.shape[4]

    cdef int n_bl = compute_half_baselines(n_ant)
    cdef int num_threads = cubical.kernels.num_omp_threads

    with nogil, parallel(num_threads=num_threads):
        for ibl in prange(n_bl, schedule='static'):
            aa, ab = half_baselines[ibl][0], half_baselines[ibl][1]
            for i in xrange(n_mod):
                for t in xrange(n_tim):
                    rr = t/t_int
                    for f in xrange(n_fre):
                        rc = f/f_int
                        for d in xrange(n_dir):
                            subtract_gmgh_product(&r[i,t,f,aa,ab,0,0], &g[d,rr,rc,aa,0,0], &m[d,i,t,f,aa,ab,0,0], &gh[d,rr,rc,ab,0,0])
                        mat_conjugate(&r[i,t,f,ab,aa,0,0], &r[i,t,f,aa,ab,0,0])

@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def cycompute_residual_1(complex3264 [:,:,:,:,:,:,:,:] m,
                       complex3264 [:,:,:,:,:,:] g,
                       complex3264 [:,:,:,:,:,:] gh,
                       complex3264 [:,:,:,:,:,:,:] r,
                       int t_int,
                       int f_int):

    """
    Given the model, gains, and their conjugates, computes the residual. Residual has full time and
    frequency resolution - solution intervals are used to correctly associate the gains with the
    model.

    Args:
        m (np.complex64 or np.complex128):
            Typed memoryview of model array with dimensions (d, m, t, f, a, a, c, c).
        g (np.complex64 or np.complex128):
            Typed memoryview of gain array with dimension (d, ti, fi, a, c, c).
        gh (np.complex64 or np.complex128):
            Typed memoryview of conjugate gain array with dimensions (d, ti, fi, a, c, c).
        r (np.complex64 or np.complex128):
            Typed memoryview of residual array with dimensions (m, t, f, a, a, c, c).
        t_int (int):
            Number of time slots per solution interval.
        f_int (int):
            Number of frequencies per solution interval.
    """

    cdef int d, i, t, f, aa, ab, ibl, rr, rc = 0
    cdef int n_dir, n_mod, n_tim, n_fre, n_ant

    n_dir = m.shape[0]
    n_mod = m.shape[1]
    n_tim = m.shape[2]
    n_fre = m.shape[3]
    n_ant = m.shape[4]

    cdef int n_bl = compute_half_baselines(n_ant)
    cdef int num_threads = cubical.kernels.num_omp_threads

    for i in xrange(n_mod):
        for t in xrange(n_tim):
            rr = t/t_int
            for f in xrange(n_fre):
                rc = f/f_int
                with nogil, parallel(num_threads=num_threads):
                    for ibl in prange(n_bl, schedule='static'):
                        aa, ab = half_baselines[ibl][0], half_baselines[ibl][1]
                        for d in xrange(n_dir):
                            subtract_gmgh_product(&r[i,t,f,aa,ab,0,0], &g[d,rr,rc,aa,0,0], &m[d,i,t,f,aa,ab,0,0], &gh[d,rr,rc,ab,0,0])
                        mat_conjugate(&r[i,t,f,ab,aa,0,0], &r[i,t,f,aa,ab,0,0])



@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def cycompute_jh(complex3264 [:,:,:,:,:,:,:,:] m,
                 complex3264 [:,:,:,:,:,:] g,
                 complex3264 [:,:,:,:,:,:,:,:] jh,
                 int t_int,
                 int f_int):

    """
    Given the model and gains, computes the non-zero elements of J\ :sup:`H`. J\ :sup:`H` has full 
    time and frequency resolution - solution intervals are used to correctly associate the gains 
    with the model. The result here contains the useful elements of J\ :sup:`H` but does not look 
    like the analytic solution.   

    Args:
        m (np.complex64 or np.complex128):
            Typed memoryview of model array with dimensions (d, m, t, f, a, a, c, c).
        g (np.complex64 or np.complex128):
            Typed memoryview of gain array with dimension (d, ti, fi, a, c, c).
        jh (np.complex64 or np.complex128):
            Typed memoryview of J\ :sup:`H` array with dimensions (d, m, t, f, a, a, c, c).
        t_int (int):
            Number of time slots per solution interval.
        f_int (int):
            Number of frequencies per solution interval.  
    """

    cdef int d, i, t, f, aa, ab, rr, rc, gd = 0
    cdef int n_dir, n_mod, n_tim, n_fre, n_ant, g_dir

    n_dir = m.shape[0]
    n_mod = m.shape[1]
    n_tim = m.shape[2]
    n_fre = m.shape[3]
    n_ant = m.shape[4]

    g_dir = g.shape[0]

    cdef int ibl, n_bl = compute_all_baselines(n_ant)
    cdef int num_threads = cubical.kernels.num_omp_threads

    with nogil, parallel(num_threads=num_threads):
        for ibl in prange(n_bl, schedule='static'):
            aa, ab = all_baselines[ibl][0], all_baselines[ibl][1]
            for i in xrange(n_mod):
                for t in xrange(n_tim):
                    rr = t/t_int
                    for f in xrange(n_fre):
                        rc = f/f_int
                        for d in xrange(n_dir):
                            gd = d%g_dir
                            gm_product(&jh[d,i,t,f,aa,ab,0,0], &g[gd,rr,rc,aa,0,0], &m[d,i,t,f,aa,ab,0,0])

@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def cycompute_jhr(complex3264 [:,:,:,:,:,:,:,:] jh,
                  complex3264 [:,:,:,:,:,:,:] r,
                  complex3264 [:,:,:,:,:,:] jhr,
                  int t_int,
                  int f_int):

    """
    Given J\ :sup:`H` and the residual (or observed data, in special cases), computes J\ :sup:`H`\R.
    J\ :sup:`H`\R is computed over intervals. 

    Args:
        jh (np.complex64 or np.complex128):
            Typed memoryview of J\ :sup:`H` array with dimensions (d, m, t, f, a, a, c, c).
        r (np.complex64 or np.complex128):
            Typed memoryview of residual array with dimensions (m, t, f, a, a, c, c).
        jhr (np.complex64 or np.complex128):
            Typed memoryview of J\ :sup:`H`\R array with dimensions (d, ti, fi, a, c, c).
        t_int (int):
            Number of time slots per solution interval.
        f_int (int):
            Number of frequencies per solution interval.  
    """

    cdef int d, i, t, f, aa, ab, rr, rc = 0
    cdef int n_dir, n_mod, n_tim, n_fre, n_ant

    n_dir = jh.shape[0]
    n_mod = jh.shape[1]
    n_tim = jh.shape[2]
    n_fre = jh.shape[3]
    n_ant = jh.shape[4]

    cdef int num_threads = cubical.kernels.num_omp_threads

    with nogil, parallel(num_threads=num_threads):
        for aa in prange(n_ant, schedule='static'):
            for ab in xrange(n_ant):
                for i in xrange(n_mod):
                    for t in xrange(n_tim):
                        rr = t/t_int
                        for f in xrange(n_fre):
                            rc = f/f_int
                            for d in xrange(n_dir):
                                add_rjh_product(&jhr[d,rr,rc,aa,0,0], &r[i,t,f,aa,ab,0,0], &jh[d,i,t,f,ab,aa,0,0])



@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def cycompute_jhj(complex3264 [:,:,:,:,:,:,:,:] jh,
                  complex3264 [:,:,:,:,:,:] jhj,
                  int t_int,
                  int f_int):
    """
    Given J\ :sup:`H` ,computes the diagonal entries of J\ :sup:`H`\J. J\ :sup:`H`\J is computed
    over intervals. This is an approximation of the Hessian.

    Args:
        jh (np.complex64 or np.complex128):
            Typed memoryview of J\ :sup:`H` array with dimensions (d, m, t, f, a, a, c, c).
        jhj (np.complex64 or np.complex128):
            Typed memoryview of J\ :sup:`H`\J array with dimensions (d, ti, fi, a, c, c).
        t_int (int):
            Number of time slots per solution interval.
        f_int (int):
            Number of frequencies per solution interval.
    """

    cdef int d, i, t, f, aa, ab, rr, rc = 0
    cdef int n_dir, n_mod, n_tim, n_fre, n_ant

    n_dir = jh.shape[0]
    n_mod = jh.shape[1]
    n_tim = jh.shape[2]
    n_fre = jh.shape[3]
    n_ant = jh.shape[4]

    cdef int num_threads = cubical.kernels.num_omp_threads

    with nogil, parallel(num_threads=num_threads):
        for aa in prange(n_ant, schedule='static'):
            for ab in xrange(n_ant):
                for i in xrange(n_mod):
                    for t in xrange(n_tim):
                        rr = t/t_int
                        for f in xrange(n_fre):
                            rc = f/f_int
                            for d in xrange(n_dir):
                                add_jhj_product(&jhj[d,rr,rc,aa,0,0], &jh[d,i,t,f,ab,aa,0,0])


@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def cycompute_jhjinv(complex3264 [:,:,:,:,:,:] jhj,
                     complex3264 [:,:,:,:,:,:] jhjinv,
                     np.uint16_t [:,:,:,:] flags,
                     float eps,
                     int flagbit):
    """
    Given J\ :sup:`H`\J (or an array with similar dimensions), computes its inverse. Takes flags
    into account and will flag additional visibilities if the inverse is too large.  

    Args:
        jhj (np.complex64 or np.complex128):
            Typed memoryview of J\ :sup:`H`\J array with dimensions (d, ti, fi, a, c, c).
        jhjinv (np.complex64 or np.complex128):
            Typed memoryview of (J\ :sup:`H`\J)\ :sup:`-1` array with dimensions 
            (d, ti, fi, a, c, c).
        flags (np.uint16_t):
            Typed memoryview of flag array with dimensions (d, t, f, a).
        eps (float):
            Threshold beneath which the denominator is regarded as too small for inversion.
        flagbit (int):
            The bitflag which will be raised if flagging is required.

    Returns:
        int:
            Number of visibilities flagged.
    """

    cdef int d, t, f, aa, ab = 0
    cdef int n_dir, n_tim, n_fre, n_ant
    cdef complex3264 denom = 0
    cdef int flag_count = 0

    eps = eps**2

    n_dir = jhj.shape[0]
    n_tim = jhj.shape[1]
    n_fre = jhj.shape[2]
    n_ant = jhj.shape[3]

    cdef int num_threads = cubical.kernels.num_omp_threads
    with nogil, parallel(num_threads=num_threads):
        for aa in prange(n_ant, schedule='static'):
            for t in xrange(n_tim):
                for f in xrange(n_fre):
                    for d in xrange(n_dir):
                        if flags[d,t,f,aa]:

                                jhjinv[d,t,f,aa,0,0] = 0
                                jhjinv[d,t,f,aa,1,1] = 0
                                jhjinv[d,t,f,aa,0,1] = 0
                                jhjinv[d,t,f,aa,1,0] = 0

                        else:
                            denom = jhj[d,t,f,aa,0,0] * jhj[d,t,f,aa,1,1] - \
                                    jhj[d,t,f,aa,0,1] * jhj[d,t,f,aa,1,0]

                            if (denom*denom.conjugate()).real<=eps:

                                jhjinv[d,t,f,aa,0,0] = 0
                                jhjinv[d,t,f,aa,1,1] = 0
                                jhjinv[d,t,f,aa,0,1] = 0
                                jhjinv[d,t,f,aa,1,0] = 0

                                flags[d,t,f,aa] = flagbit
                                flag_count += 1

                            else:

                                jhjinv[d,t,f,aa,0,0] = jhj[d,t,f,aa,1,1]/denom
                                jhjinv[d,t,f,aa,1,1] = jhj[d,t,f,aa,0,0]/denom
                                jhjinv[d,t,f,aa,0,1] = -1 * jhj[d,t,f,aa,0,1]/denom
                                jhjinv[d,t,f,aa,1,0] = -1 * jhj[d,t,f,aa,1,0]/denom

    return flag_count


@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def cycompute_update(complex3264 [:,:,:,:,:,:] jhr,
                     complex3264 [:,:,:,:,:,:] jhjinv,
                     complex3264 [:,:,:,:,:,:] upd):
    """
    Given J\ :sup:`H`\R and (J\ :sup:`H`\J)\ :sup:`-1`, computes the gain update. The dimensions of
    the input should already be consistent, making this operation simple.

    Args:
        jhr (np.complex64 or np.complex128):
            Typed memoryview of J\ :sup:`H`\R array with dimensions (d, ti, fi, a, c, c).
        jhjinv (np.complex64 or np.complex128):
            Typed memoryview of (J\ :sup:`H`\J)\ :sup:`-1` array with dimensions 
            (d, ti, fi, a, c, c).
        upd (np.complex64 or np.complex128):
            Typed memoryview of gain update array with dimensions (d, ti, fi, a, c, c).
    """

    cdef int d, t, f, aa = 0
    cdef int n_dir, n_tim, n_fre, n_ant

    n_dir = jhr.shape[0]
    n_tim = jhr.shape[1]
    n_fre = jhr.shape[2]
    n_ant = jhr.shape[3]

    cdef int num_threads = cubical.kernels.num_omp_threads
    with nogil, parallel(num_threads=num_threads):
        for aa in prange(n_ant, schedule='static'):
            for t in xrange(n_tim):
                for f in xrange(n_fre):
                    for d in xrange(n_dir):
                        mat_product(&upd[d,t,f,aa,0,0], &jhr[d,t,f,aa,0,0], &jhjinv[d,t,f,aa,0,0])



@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def cycompute_corrected(complex3264 [:,:,:,:,:,:] o,
                        complex3264 [:,:,:,:,:,:] g,
                        complex3264 [:,:,:,:,:,:] gh,
                        complex3264 [:,:,:,:,:,:] corr,
                        int t_int,
                        int f_int):

    """
    Given the observed visbilities, inverse gains, and their conjugates, computes the corrected 
    visibilitites.  

    Args:
        o (np.complex64 or np.complex128):
            Typed memoryview of observed visibility array with dimensions (t, f, a, a, c, c).
        g (np.complex64 or np.complex128):
            Typed memoryview of inverse gain array with dimensions (d, ti, fi, a, c, c).
        gh (np.complex64 or np.complex128):
            Typed memoryview of inverse conjugate gain array with dimensions (d, ti, fi, a, c, c).
        corr (np.complex64 or np.complex128):
            Typed memoryview of corrected data array with dimensions (t, f, a, a, c, c).
        t_int (int):
            Number of time slots per solution interval.
        f_int (int):
            Number of frequencies per solution interval.  
    """

    cdef int d, t, f, aa, ab, rr, rc = 0
    cdef int n_dir, n_tim, n_fre, n_ant

    n_dir = g.shape[0]
    n_tim = o.shape[0]
    n_fre = o.shape[1]
    n_ant = o.shape[2]

    cdef int ibl, n_bl = compute_half_baselines(n_ant)
    cdef int num_threads = cubical.kernels.num_omp_threads

    with nogil, parallel(num_threads=num_threads):
        for ibl in prange(n_bl, schedule='static'):
            aa, ab = half_baselines[ibl][0], half_baselines[ibl][1]
            for t in xrange(n_tim):
                rr = t/t_int
                for f in xrange(n_fre):
                    rc = f/f_int
                    gmgh_product(&corr[t,f,aa,ab,0,0], &g[0,rr,rc,aa,0,0], &o[t,f,aa,ab,0,0], &gh[0,rr,rc,ab,0,0])
                    mat_conjugate(&corr[t,f,ab,aa,0,0], &corr[t,f,aa,ab,0,0])

@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def cyapply_gains_slow(complex3264 [:,:,:,:,:,:,:,:] m,
                  complex3264 [:,:,:,:,:,:] g,
                  complex3264 [:,:,:,:,:,:] gh,
                  int t_int,
                  int f_int):

    """
    Applies the gains and their cinjugates to the model array. This operation is performed in place
    - be wary of losing the original array. The result has full time and frequency resolution - 
    solution intervals are used to correctly associate the gains with the model.

    Args:
        m (np.complex64 or np.complex128):
            Typed memoryview of model visibility array with dimensions (d, m , t, f, a, a, c, c).
        g (np.complex64 or np.complex128):
            Typed memoryview of gain array with dimensions (d, ti, fi, a, c, c).
        gh (np.complex64 or np.complex128):
            Typed memoryview of conjugate gain array with dimensions (d, ti, fi, a, c, c).
        t_int (int):
            Number of time slots per solution interval.
        f_int (int):
            Number of frequencies per solution interval.

    For reasons unknown, this ordering of the loops leads to slower performance...
    """

    cdef int d, i, t, f, aa, ab, rr, rc, gd = 0
    cdef int n_dir, n_mod, n_tim, n_fre, n_ant, g_dir
    cdef complex3264 gmtmp1, gmtmp2, gmtmp3, gmtmp4

    n_dir = m.shape[0]
    n_mod = m.shape[1]
    n_tim = m.shape[2]
    n_fre = m.shape[3]
    n_ant = m.shape[4]

    g_dir = g.shape[0]

    cdef int ibl, n_bl = compute_half_baselines(n_ant)
    cdef int num_threads = cubical.kernels.num_omp_threads

    with nogil, parallel(num_threads=num_threads):
        for ibl in prange(n_bl, schedule='static'):
            aa, ab = half_baselines[ibl][0], half_baselines[ibl][1]
            for i in xrange(n_mod):
                for t in xrange(n_tim):
                    rr = t/t_int
                    for f in xrange(n_fre):
                        rc = f/f_int
                        for d in xrange(n_dir):
                            gd = d%g_dir
                            inplace_gmgh_product(&g[d,rr,rc,aa,0,0], &m[d,i,t,f,aa,ab,0,0], &gh[d,rr,rc,ab,0,0])
                            mat_conjugate(&m[d,i,t,f,ab,aa,0,0], &m[d,i,t,f,aa,ab,0,0])

@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def cyapply_gains(complex3264 [:,:,:,:,:,:,:,:] m,
                  complex3264 [:,:,:,:,:,:] g,
                  complex3264 [:,:,:,:,:,:] gh,
                  int t_int,
                  int f_int):

    """
    Applies the gains and their cinjugates to the model array. This operation is performed in place
    - be wary of losing the original array. The result has full time and frequency resolution -
    solution intervals are used to correctly associate the gains with the model.

    Args:
        m (np.complex64 or np.complex128):
            Typed memoryview of model visibility array with dimensions (d, m , t, f, a, a, c, c).
        g (np.complex64 or np.complex128):
            Typed memoryview of gain array with dimensions (d, ti, fi, a, c, c).
        gh (np.complex64 or np.complex128):
            Typed memoryview of conjugate gain array with dimensions (d, ti, fi, a, c, c).
        t_int (int):
            Number of time slots per solution interval.
        f_int (int):
            Number of frequencies per solution interval.

    For reasons unknown, this ordering of the loops leads to faster performance...
    """

    cdef int d, i, t, f, aa, ab, rr, rc, gd = 0
    cdef int n_dir, n_mod, n_tim, n_fre, n_ant, g_dir
    cdef complex3264 gmtmp1, gmtmp2, gmtmp3, gmtmp4

    n_dir = m.shape[0]
    n_mod = m.shape[1]
    n_tim = m.shape[2]
    n_fre = m.shape[3]
    n_ant = m.shape[4]

    g_dir = g.shape[0]

    cdef int ibl, n_bl = compute_half_baselines(n_ant)
    cdef int num_threads = cubical.kernels.num_omp_threads

    for d in xrange(n_dir):
        gd = d%g_dir
        for i in xrange(n_mod):
            for t in xrange(n_tim):
                rr = t/t_int
                for f in xrange(n_fre):
                    rc = f/f_int
                    with nogil, parallel(num_threads=num_threads):
                        for ibl in prange(n_bl, schedule='static'):
                            aa, ab = half_baselines[ibl][0], half_baselines[ibl][1]
                            inplace_gmgh_product(&g[d,rr,rc,aa,0,0], &m[d,i,t,f,aa,ab,0,0], &gh[d,rr,rc,ab,0,0])
                            mat_conjugate(&m[d,i,t,f,ab,aa,0,0], &m[d,i,t,f,aa,ab,0,0])

@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def cyapply_gains_1(complex3264 [:,:,:,:,:,:,:,:] m,
                  complex3264 [:,:,:,:,:,:] g,
                  complex3264 [:,:,:,:,:,:] gh,
                  int t_int,
                  int f_int):

    """
    Applies the gains and their cinjugates to the model array. This operation is performed in place
    - be wary of losing the original array. The result has full time and frequency resolution -
    solution intervals are used to correctly associate the gains with the model.

    Args:
        m (np.complex64 or np.complex128):
            Typed memoryview of model visibility array with dimensions (d, m , t, f, a, a, c, c).
        g (np.complex64 or np.complex128):
            Typed memoryview of gain array with dimensions (d, ti, fi, a, c, c).
        gh (np.complex64 or np.complex128):
            Typed memoryview of conjugate gain array with dimensions (d, ti, fi, a, c, c).
        t_int (int):
            Number of time slots per solution interval.
        f_int (int):
            Number of frequencies per solution interval.

    For reasons unknown, this ordering of the loops leads to faster performance...
    """

    cdef int d, i, t, f, aa, ab, rr, rc, gd = 0
    cdef int n_dir, n_mod, n_tim, n_fre, n_ant, g_dir
    cdef complex3264 gmtmp1, gmtmp2, gmtmp3, gmtmp4

    n_dir = m.shape[0]
    n_mod = m.shape[1]
    n_tim = m.shape[2]
    n_fre = m.shape[3]
    n_ant = m.shape[4]

    g_dir = g.shape[0]

    cdef long ibl, n_bl = compute_half_baselines(n_ant)
    cdef long num_threads = cubical.kernels.num_omp_threads
    cdef long nbf = n_bl*n_fre, nbtf = nbf*n_tim, nbtfm=nbtf*n_mod, nbtfmd = nbtfm*n_dir, loop, j

    with nogil, parallel(num_threads=num_threads):
        for loop in prange(nbtfmd, schedule='static'):
            ibl = loop%n_bl
            j = loop/n_bl
            f = j%n_fre
            j = j/n_fre
            t = j%n_tim
            j = j/n_tim
            i = j%n_mod
            d = j/n_mod
            rr = t/t_int
            rc = f/f_int
            aa, ab = half_baselines[ibl][0], half_baselines[ibl][1]
            inplace_gmgh_product(&g[d,rr,rc,aa,0,0], &m[d,i,t,f,aa,ab,0,0], &gh[d,rr,rc,ab,0,0])
            mat_conjugate(&m[d,i,t,f,ab,aa,0,0], &m[d,i,t,f,aa,ab,0,0])


