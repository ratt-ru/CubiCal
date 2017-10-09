"""
Cython kernels for the Jones chain machine. Functions require output arrays to be 
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

from cython.parallel import prange, parallel
import numpy as np
cimport numpy as np
import cython

ctypedef fused complex3264:
    np.complex64_t
    np.complex128_t

@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def cycompute_jh(complex3264 [:,:,:,:,:,:,:,:] jh,
                 complex3264 [:,:,:,:,:,:] g,                 
                 int t_int,
                 int f_int):

    """
    Given J\ :sup:`H` (initially populated with the model array) and gains, computes the non-zero 
    elements of J\ :sup:`H`. J\ :sup:`H` has full time and frequency resolution - solution intervals
    are used to correctly associate the gains with the model. The result here contains the useful 
    elements of J\ :sup:`H` but does not look like the analytic solution. This function is called
    multiple times during a chain computation. Each call applies a different gain term.   

    Args:
        jh (np.complex64 or np.complex128):
            Typed memoryview of J\ :sup:`H` array with dimensions (d, m, t, f, a, a, c, c).
        g (np.complex64 or np.complex128):
            Typed memoryview of gain array with dimension (d, ti, fi, a, c, c).
        t_int (int):
            Number of time slots per solution interval.
        f_int (int):
            Number of frequencies per solution interval.
    """

    cdef int d, i, t, f, aa, ab, rr, rc, gd = 0
    cdef int n_dir, n_mod, n_tim, n_fre, n_ant, g_dir
    cdef complex3264 jh00, jh01, jh10, jh11

    n_dir = jh.shape[0]
    n_mod = jh.shape[1]
    n_tim = jh.shape[2]
    n_fre = jh.shape[3]
    n_ant = jh.shape[4]

    g_dir = g.shape[0]

    for d in xrange(n_dir):
        gd = d%g_dir
        for i in xrange(n_mod):
            for t in xrange(n_tim):
                rr = t/t_int
                for f in xrange(n_fre):
                    rc = f/f_int
                    for aa in xrange(n_ant):
                        for ab in xrange(n_ant):

                            jh00 = jh[d,i,t,f,aa,ab,0,0]
                            jh10 = jh[d,i,t,f,aa,ab,1,0]
                            jh01 = jh[d,i,t,f,aa,ab,0,1]
                            jh11 = jh[d,i,t,f,aa,ab,1,1]


                            jh[d,i,t,f,aa,ab,0,0] = g[gd,rr,rc,aa,0,0]*jh00 + \
                                                    g[gd,rr,rc,aa,0,1]*jh10

                            jh[d,i,t,f,aa,ab,0,1] = g[gd,rr,rc,aa,0,0]*jh01 + \
                                                    g[gd,rr,rc,aa,0,1]*jh11

                            jh[d,i,t,f,aa,ab,1,0] = g[gd,rr,rc,aa,1,0]*jh00 + \
                                                    g[gd,rr,rc,aa,1,1]*jh10

                            jh[d,i,t,f,aa,ab,1,1] = g[gd,rr,rc,aa,1,0]*jh01 + \
                                                    g[gd,rr,rc,aa,1,1]*jh11



@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def cyapply_left_inv_jones(complex3264 [:,:,:,:,:,:] jhr,
                           complex3264 [:,:,:,:,:,:] ginv,
                           int t_int,
                           int f_int):

    """
    Applies the inverse of a gain array the left side of J\ :sup:`H`\R. J\ :sup:`H`\R has full time 
    and frequency resolution - solution intervals are used to correctly associate the gains with 
    the model. This is a special function which is unique to the Jones chain.    

    Args:
        jhr (np.complex64 or np.complex128):
            Typed memoryview of J\ :sup:`H`\R array with dimensions (d, t, f, a, c, c).
        ginv (np.complex64 or np.complex128):
            Typed memoryview of inverse gain array with dimensions (d, ti, fi, a, c, c).
        t_int (int):
            Number of time slots per solution interval.
        f_int (int):
            Number of frequencies per solution interval.
    """

    cdef int d, i, t, f, aa, ab, rr, rc, gd = 0
    cdef int n_dir, n_tim, n_fre, n_ant, g_dir
    cdef complex3264 jhr00, jhr01, jhr10, jhr11

    n_dir = jhr.shape[0]
    n_tim = jhr.shape[1]
    n_fre = jhr.shape[2]
    n_ant = jhr.shape[3]

    g_dir = ginv.shape[0]

    for d in xrange(n_dir):
        gd = d%g_dir
        for t in xrange(n_tim):
            rr = t/t_int
            for f in xrange(n_fre):
                rc = f/f_int
                for aa in xrange(n_ant):

                    jhr00 = jhr[d,t,f,aa,0,0]
                    jhr01 = jhr[d,t,f,aa,0,1]
                    jhr10 = jhr[d,t,f,aa,1,0]
                    jhr11 = jhr[d,t,f,aa,1,1]

                    jhr[d,t,f,aa,0,0] = ginv[gd,rr,rc,aa,0,0]*jhr00 + \
                                        ginv[gd,rr,rc,aa,0,1]*jhr10

                    jhr[d,t,f,aa,0,1] = ginv[gd,rr,rc,aa,0,0]*jhr01 + \
                                        ginv[gd,rr,rc,aa,0,1]*jhr11

                    jhr[d,t,f,aa,1,0] = ginv[gd,rr,rc,aa,1,0]*jhr00 + \
                                        ginv[gd,rr,rc,aa,1,1]*jhr10

                    jhr[d,t,f,aa,1,1] = ginv[gd,rr,rc,aa,1,0]*jhr01 + \
                                        ginv[gd,rr,rc,aa,1,1]*jhr11

@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def cysum_jhr_intervals(complex3264 [:,:,:,:,:,:] jhr,
                        complex3264 [:,:,:,:,:,:] jhrint,
                        int t_int,
                        int f_int):

    """
    Collapses J\ :sup:`H`\R to be cosistent with the solution intervals for the current gain of 
    interest. This is necessary as each gain term in a chain may have unique solution intervals. 

    Args:
        jhr (np.complex64 or np.complex128):
            Typed memoryview of J\ :sup:`H`\R array with dimensions (d, t, f, a, c, c).
        jhrint (np.complex64 or np.complex128):
            Typed memoryview of collapsed J\ :sup:`H`\R array with dimensions (d, ti, fi, a, c, c).
        t_int (int):
            Number of time slots per solution interval.
        f_int (int):
            Number of frequencies per solution interval.
    """

    cdef int d, i, t, f, aa, ab, rr, rc = 0
    cdef int n_dir, n_tim, n_fre, n_ant

    n_dir = jhr.shape[0]
    n_tim = jhr.shape[1]
    n_fre = jhr.shape[2]
    n_ant = jhr.shape[3]

    for d in xrange(n_dir):
        for t in xrange(n_tim):
            rr = t/t_int
            for f in xrange(n_fre):
                rc = f/f_int
                for aa in xrange(n_ant):

                    jhrint[d,rr,rc,aa,0,0] = jhrint[d,rr,rc,aa,0,0] + jhr[d,t,f,aa,0,0]

                    jhrint[d,rr,rc,aa,0,1] = jhrint[d,rr,rc,aa,0,1] + jhr[d,t,f,aa,0,1]

                    jhrint[d,rr,rc,aa,1,0] = jhrint[d,rr,rc,aa,1,0] + jhr[d,t,f,aa,1,0]

                    jhrint[d,rr,rc,aa,1,1] = jhrint[d,rr,rc,aa,1,1] + jhr[d,t,f,aa,1,1]

@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def cycompute_residual(complex3264 [:,:,:,:,:,:,:,:] m,
                       complex3264 [:,:,:,:,:,:,:] r):

    """
    Given the model array (already multipled by the gains) and the residual array (already populated
    with the observed data), computes the final residual. This is a special case where the gains
    are applied to the model elsewhere.

    Args:
        m (np.complex64 or np.complex128):
            Typed memoryview of the model array with dimensions (d, m, t, f, a, a, c, c).
        r (np.complex64 or np.complex128):
            Typed memoryview of tje residual array with dimensions (m, t, f, a, a, c, c).
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
                            r[i,t,f,aa,ab,0,0] = r[i,t,f,aa,ab,0,0] - m[d,i,t,f,aa,ab,0,0]

                            r[i,t,f,aa,ab,0,1] = r[i,t,f,aa,ab,0,1] - m[d,i,t,f,aa,ab,0,1]

                            r[i,t,f,aa,ab,1,0] = r[i,t,f,aa,ab,1,0] - m[d,i,t,f,aa,ab,1,0]

                            r[i,t,f,aa,ab,1,1] = r[i,t,f,aa,ab,1,1] - m[d,i,t,f,aa,ab,1,1]