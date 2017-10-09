"""
Cython kernels for the frequency slope (delay) gain machine. Functions require output arrays to be 
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
| Block          |   b  |
+----------------+------+
| Correlation    |   c  |
+----------------+------+

"""

from cython.parallel import prange, parallel
import numpy as np
cimport numpy as np
import cython

ctypedef fused complex3264:
    np.complex64_t
    np.complex128_t

ctypedef fused float3264:
    np.float32_t
    np.float64_t

cdef extern from "complex.h":
    double complex exp(double complex)

@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def cycompute_tmp_jhj(complex3264 [:,:,:,:,:,:,:,:] m,
                      complex3264 [:,:,:,:,:,:,:] tmp_jhj):

    """
    Given the model array, computes an intermediary result required in the computation of the 
    diagonal entries of J\ :sup:`H`\J. This is necessary as J\ :sup:`H`\J is blockwise diagonal.

    Args:
        m (np.complex64 or np.complex128):
            Typed memoryview of model array with dimensions (d, m, t, f, a, a, c, c).
        tmp_jhj (np.complex64 or np.complex128):
            Typed memoryview of J\ :sup:`H`\J array with dimensions (d, m, t, f, a, c, c).
    """

    cdef int d, i, t, f, aa, ab = 0
    cdef int n_dir, n_mod, n_tim, n_fre, n_ant

    n_dir = m.shape[0]
    n_mod = m.shape[1]
    n_tim = m.shape[2]
    n_fre = m.shape[3]
    n_ant = m.shape[4]

    for d in xrange(n_dir):
        for i in xrange(n_mod):
            for t in xrange(n_tim):
                for f in xrange(n_fre):
                    for aa in xrange(n_ant):
                        for ab in xrange(n_ant):
                            
                            tmp_jhj[d,i,t,f,aa,0,0] = tmp_jhj[d,i,t,f,aa,0,0] + \
                                                  m[d,i,t,f,aa,ab,0,0]*m[d,i,t,f,ab,aa,0,0] + \
                                                  m[d,i,t,f,aa,ab,0,1]*m[d,i,t,f,ab,aa,1,0]
                            tmp_jhj[d,i,t,f,aa,1,1] = tmp_jhj[d,i,t,f,aa,1,1] + \
                                                  m[d,i,t,f,aa,ab,1,0]*m[d,i,t,f,ab,aa,0,1] + \
                                                  m[d,i,t,f,aa,ab,1,1]*m[d,i,t,f,ab,aa,1,1]

@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def cycompute_jhj(float3264 [:,:,:,:,:,:,:] tmp_jhj,
                  float3264 [:,:,:,:,:,:,:] jhj,
                  float3264 [:] fs,
                  int t_int,
                  int f_int):

    """
    Given the intermediary J\ :sup:`H`\J and channel frequencies, computes the diagonal entries of 
    J\ :sup:`H`\J. J\ :sup:`H`\J is computed over intervals. This is an approximation of the 
    Hessian. In the phase-only frequency slope case, this approximation does not depend on the 
    gains, therefore it does not vary with iteration. The addition of the block dimension is
    required due to the block-wise diagonal nature of J\ :sup:`H`\J.

    Args:
        tmp_jhj (np.float32 or np.float64):
            Typed memoryview of intermiadiary J\ :sup:`H`\J array with dimensions 
            (d, m, t, f, a, c, c).
        jhj (np.float32 or np.float64):
            Typed memoryview of J\ :sup:`H`\J array with dimensions (d, ti, fi, a, b, c, c).
        fs (np.float32 or np.float64):
            Typed memoryview of channel frequencies with dimension (f).
        t_int (int):
            Number of time slots per solution interval.
        f_int (int):
            Number of frequencies per solution interval.  
    """

    cdef int d, i, t, f, aa, ab = 0
    cdef int n_dir, n_mod, n_tim, n_fre, n_ant

    n_dir = tmp_jhj.shape[0]
    n_mod = tmp_jhj.shape[1]
    n_tim = tmp_jhj.shape[2]
    n_fre = tmp_jhj.shape[3]
    n_ant = tmp_jhj.shape[4]

    for d in xrange(n_dir):
        for i in xrange(n_mod):
            for t in xrange(n_tim):
                rr = t/t_int
                for f in xrange(n_fre):
                    rc = f/f_int
                    for aa in xrange(n_ant):
                        for ab in xrange(n_ant): 

                            jhj[d,rr,rc,aa,0,0,0] = jhj[d,rr,rc,aa,0,0,0] + \
                                                    fs[f]*fs[f]*tmp_jhj[d,i,t,f,aa,0,0]
                            jhj[d,rr,rc,aa,0,1,1] = jhj[d,rr,rc,aa,0,1,1] + \
                                                    fs[f]*fs[f]*tmp_jhj[d,i,t,f,aa,1,1]

                            jhj[d,rr,rc,aa,1,0,0] = jhj[d,rr,rc,aa,1,0,0] + \
                                                    fs[f]*tmp_jhj[d,i,t,f,aa,0,0]
                            jhj[d,rr,rc,aa,1,1,1] = jhj[d,rr,rc,aa,1,1,1] + \
                                                    fs[f]*tmp_jhj[d,i,t,f,aa,1,1]

                            jhj[d,rr,rc,aa,2,0,0] = jhj[d,rr,rc,aa,2,0,0] + \
                                                    tmp_jhj[d,i,t,f,aa,0,0]
                            jhj[d,rr,rc,aa,2,1,1] = jhj[d,rr,rc,aa,2,1,1] + \
                                                    tmp_jhj[d,i,t,f,aa,1,1]



@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def cycompute_jhjinv(float3264 [:,:,:,:,:,:,:] jhj,
                     float3264 [:,:,:,:,:,:,:] jhjinv,
                     float eps):
    """
    Given J\ :sup:`H`\J (or an array with similar dimensions), computes its inverse.  

    Args:
        jhj (np.float32 or np.float64):
            Typed memoryview of J\ :sup:`H`\J array with dimensions (d, ti, fi, a, b, c, c).
        jhjinv (np.float32 or np.float64):
            Typed memoryview of (J\ :sup:`H`\J)\ :sup:`-1` array with dimensions 
            (d, ti, fi, a, b, c, c).
        eps (float):
            Threshold beneath which the denominator is regarded as too small for inversion.
    """

    cdef int d, t, f, c, aa = 0
    cdef int n_dir, n_tim, n_fre, n_ant

    n_dir = jhj.shape[0]
    n_tim = jhj.shape[1]
    n_fre = jhj.shape[2]
    n_ant = jhj.shape[3]

    for d in xrange(n_dir):
        for t in xrange(n_tim):
            for f in xrange(n_fre):
                for aa in xrange(n_ant):
                    for c in xrange(2):
                                                     
                        det = jhj[d,t,f,aa,0,c,c]*jhj[d,t,f,aa,2,c,c] - \
                              jhj[d,t,f,aa,1,c,c]*jhj[d,t,f,aa,1,c,c] 

                        if det<eps:

                            jhjinv[d,t,f,aa,0,c,c] = 0
                            jhjinv[d,t,f,aa,1,c,c] = 0
                            jhjinv[d,t,f,aa,2,c,c] = 0

                        else:

                            det = 1/det

                            jhjinv[d,t,f,aa,0,c,c] = det*jhj[d,t,f,aa,2,c,c]

                            jhjinv[d,t,f,aa,1,c,c] = -1*det*jhj[d,t,f,aa,1,c,c]

                            jhjinv[d,t,f,aa,2,c,c] = det*jhj[d,t,f,aa,0,c,c]

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

    cdef int d, i, t, f, aa, ab, rr, rc = 0
    cdef int n_dir, n_mod, n_tim, n_fre, n_ant

    n_dir = m.shape[0]
    n_mod = m.shape[1]
    n_tim = m.shape[2]
    n_fre = m.shape[3]
    n_ant = m.shape[4]

    for d in xrange(n_dir):
        for i in xrange(n_mod):
            for t in xrange(n_tim):
                rr = t/t_int
                for f in xrange(n_fre):
                    rc = f/f_int
                    for aa in xrange(n_ant):
                        for ab in xrange(n_ant):
                            jh[d,i,t,f,aa,ab,0,0] = g[d,rr,rc,aa,0,0]*m[d,i,t,f,aa,ab,0,0]

                            jh[d,i,t,f,aa,ab,0,1] = g[d,rr,rc,aa,0,0]*m[d,i,t,f,aa,ab,0,1]

                            jh[d,i,t,f,aa,ab,1,0] = g[d,rr,rc,aa,1,1]*m[d,i,t,f,aa,ab,1,0]

                            jh[d,i,t,f,aa,ab,1,1] = g[d,rr,rc,aa,1,1]*m[d,i,t,f,aa,ab,1,1]

@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def cycompute_tmp_jhr(complex3264 [:,:,:,:,:,:] gh,
                      complex3264 [:,:,:,:,:,:,:,:] jh,
                      complex3264 [:,:,:,:,:,:,:] r,
                      complex3264 [:,:,:,:,:,:] jhr,
                      int t_int,
                      int f_int):

    """
    Given the conjugate gains, J\ :sup:`H` and the residual (or observed data, in special cases), 
    computes an intermediary result required for computing J\ :sup:`H`\R. This result is computed
    over intervals and is required due to the structure of J\ :sup:`H`.

    Args:
        gh (np.complex64 or np.complex128):
            Typed memoryview of gain array with dimension (d, ti, fi, a, c, c).
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

    for d in xrange(n_dir):
        for i in xrange(n_mod):
            for t in xrange(n_tim):
                rr = t/t_int
                for f in xrange(n_fre):
                    rc = f/f_int
                    for aa in xrange(n_ant):
                        for ab in xrange(n_ant):
                            jhr[d,rr,rc,aa,0,0] = jhr[d,rr,rc,aa,0,0] + gh[d,rr,rc,aa,0,0] * (
                                                      r[i,t,f,aa,ab,0,0]*jh[d,i,t,f,ab,aa,0,0] + 
                                                      r[i,t,f,aa,ab,0,1]*jh[d,i,t,f,ab,aa,1,0]   )

                            jhr[d,rr,rc,aa,1,1] = jhr[d,rr,rc,aa,1,1] + gh[d,rr,rc,aa,1,1] * (
                                                      r[i,t,f,aa,ab,1,0]*jh[d,i,t,f,ab,aa,0,1] + 
                                                      r[i,t,f,aa,ab,1,1]*jh[d,i,t,f,ab,aa,1,1]   )

@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def cycompute_jhr(float3264 [:,:,:,:,:,:] tmp_jhr,
                  float3264 [:,:,:,:,:,:,:] jhr,
                  float3264 [:] fs,
                  int t_int,
                  int f_int):

    """
    Given the intermediary J\ :sup:`H`\R and channel frequencies, computes J\ :sup:`H`\R. 
    J\ :sup:`H`\R is computed over intervals. The addition of the block dimension is
    required due to the structure of J\ :sup:`H`.

    Args:
        tmp_jhr (np.float32 or np.float64):
            Typed memoryview of intermiadiary J\ :sup:`H`\J array with dimensions 
            (d, ti, fi, a, c, c).
        jhr (np.float32 or np.float64):
            Typed memoryview of J\ :sup:`H`\J array with dimensions (d, ti, fi, a, b, c, c).
        fs (np.float32 or np.float64):
            Typed memoryview of channel frequencies with dimension (f).
        t_int (int):
            Number of time slots per solution interval.
        f_int (int):
            Number of frequencies per solution interval.   
    """

    cdef int d, i, t, f, aa, ab, c, rr, rc = 0
    cdef int n_dir, n_mod, n_tim, n_fre, n_ant

    n_dir = tmp_jhr.shape[0]
    n_tim = tmp_jhr.shape[1]
    n_fre = tmp_jhr.shape[2]
    n_ant = tmp_jhr.shape[3]

    for d in xrange(n_dir):
        for t in xrange(n_tim):
            rr = t/t_int
            for f in xrange(n_fre):
                rc = f/f_int
                for aa in xrange(n_ant):
                    for c in xrange(2):
                        jhr[d,rr,rc,aa,0,c,c] = jhr[d,rr,rc,aa,0,c,c] + fs[f]*tmp_jhr[d,t,f,aa,c,c]
                        jhr[d,rr,rc,aa,1,c,c] = jhr[d,rr,rc,aa,1,c,c] +       tmp_jhr[d,t,f,aa,c,c]
                        

@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def cycompute_update(float3264 [:,:,:,:,:,:,:] jhr,
                     float3264 [:,:,:,:,:,:,:] jhj,
                     float3264 [:,:,:,:,:,:,:] upd):
    """
    Given J\ :sup:`H`\R and (J\ :sup:`H`\J)\ :sup:`-1`, computes the gain update. The dimensions of
    the input should already be consistent, making this operation simple. These arrays are real
    valued.

    Args:
        jhr (np.float32 or np.float64):
            Typed memoryview of J\ :sup:`H`\R array with dimensions (d, ti, fi, a, b, c, c).
        jhjinv (np.float32 or np.float64):
            Typed memoryview of (J\ :sup:`H`\J)\ :sup:`-1` array with dimensions 
            (d, ti, fi, a, b, c, c).
        upd (np.float32 or np.float64):
            Typed memoryview of gain update array with dimensions (d, ti, fi, a, b, c, c).
    """

    cdef int d, t, f, aa, c = 0
    cdef int n_dir, n_tim, n_fre, n_ant

    n_dir = jhr.shape[0]
    n_tim = jhr.shape[1]
    n_fre = jhr.shape[2]
    n_ant = jhr.shape[3]

    for d in xrange(n_dir):
        for t in xrange(n_tim):
            for f in xrange(n_fre):
                for aa in xrange(n_ant):
                    for c in xrange(2): 

                        upd[d,t,f,aa,0,c,c] = jhj[d,t,f,aa,0,c,c]*jhr[d,t,f,aa,0,c,c] + \
                                              jhj[d,t,f,aa,1,c,c]*jhr[d,t,f,aa,1,c,c]

                        upd[d,t,f,aa,1,c,c] = jhj[d,t,f,aa,1,c,c]*jhr[d,t,f,aa,0,c,c] + \
                                              jhj[d,t,f,aa,2,c,c]*jhr[d,t,f,aa,1,c,c]

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

    cdef int d, i, t, f, aa, ab, rr, rc = 0
    cdef int n_dir, n_mod, n_tim, n_fre, n_ant

    n_dir = m.shape[0]
    n_mod = m.shape[1]
    n_tim = m.shape[2]
    n_fre = m.shape[3]
    n_ant = m.shape[4]

    for d in xrange(n_dir):
        for i in xrange(n_mod):
            for t in xrange(n_tim):
                rr = t/t_int
                for f in xrange(n_fre):
                    rc = f/f_int
                    for aa in xrange(n_ant):
                        for ab in xrange(n_ant):
                            r[i,t,f,aa,ab,0,0] = r[i,t,f,aa,ab,0,0] - \
                                            (g[d,rr,rc,aa,0,0]*m[d,i,t,f,aa,ab,0,0]*gh[d,rr,rc,ab,0,0])

                            r[i,t,f,aa,ab,0,1] = r[i,t,f,aa,ab,0,1] - \
                                            (g[d,rr,rc,aa,0,0]*m[d,i,t,f,aa,ab,0,1]*gh[d,rr,rc,ab,1,1])

                            r[i,t,f,aa,ab,1,0] = r[i,t,f,aa,ab,1,0] - \
                                            (g[d,rr,rc,aa,1,1]*m[d,i,t,f,aa,ab,1,0]*gh[d,rr,rc,ab,0,0])

                            r[i,t,f,aa,ab,1,1] = r[i,t,f,aa,ab,1,1] - \
                                            (g[d,rr,rc,aa,1,1]*m[d,i,t,f,aa,ab,1,1]*gh[d,rr,rc,ab,1,1])

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

    for d in xrange(n_dir):
        for t in xrange(n_tim):
            rr = t/t_int
            for f in xrange(n_fre):
                rc = f/f_int
                for aa in xrange(n_ant):
                    for ab in xrange(n_ant):
                        corr[t,f,aa,ab,0,0] = \
                        g[d,rr,rc,aa,0,0]*o[t,f,aa,ab,0,0]*gh[d,rr,rc,ab,0,0]

                        corr[t,f,aa,ab,0,1] = \
                        g[d,rr,rc,aa,0,0]*o[t,f,aa,ab,0,1]*gh[d,rr,rc,ab,1,1]

                        corr[t,f,aa,ab,1,0] = \
                        g[d,rr,rc,aa,1,1]*o[t,f,aa,ab,1,0]*gh[d,rr,rc,ab,0,0]

                        corr[t,f,aa,ab,1,1] = \
                        g[d,rr,rc,aa,1,1]*o[t,f,aa,ab,1,1]*gh[d,rr,rc,ab,1,1]


@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def cyconstruct_gains(float3264 [:,:,:,:,:,:,:] param,
                      complex3264 [:,:,:,:,:,:] g,
                      float3264 [:] fs,
                      int t_int,
                      int f_int):

    """
    Given the real-valued parameters of the gains, computes the complex gains. 

    Args:
        param (np.float32 or np.float64):
            Typed memoryview of parameter array with dimensions (d, ti, fi, a, b, c, c).
        g (np.complex64 or np.complex128):
            Typed memoryview of inverse gain array with dimensions (d, ti, fi, a, c, c).
        fs (np.float32 or np.float64):
            Typed memoryview of channel frequencies with dimension (f).
        t_int (int):
            Number of time slots per solution interval.
        f_int (int):
            Number of frequencies per solution interval.  
    """

    cdef int d, t, f, aa, rr, rc, c = 0
    cdef int n_dir, n_tim, n_fre, n_ant

    n_dir = g.shape[0]
    n_tim = g.shape[1]
    n_fre = g.shape[2]
    n_ant = g.shape[3]

    for d in xrange(n_dir):
        for t in xrange(n_tim):
            rr = t/t_int
            for f in xrange(n_fre):
                rc = f/f_int
                for aa in xrange(n_ant):
                    for c in xrange(2):
                        g[d,t,f,aa,c,c] = exp(1j*(fs[f]*param[d,rr,rc,aa,0,c,c] + 
                                                        param[d,rr,rc,aa,1,c,c] ))
