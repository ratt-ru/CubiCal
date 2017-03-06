#cython:profile=True

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
def cycompute_residual(complex3264 [:,:,:,:,:,:,:] m,
                       complex3264 [:,:,:,:,:,:] g,
                       complex3264 [:,:,:,:,:,:] gh,
                       complex3264 [:,:,:,:,:,:] o,
                       complex3264 [:,:,:,:,:,:] r,
                       int t_int,
                       int f_int):

    """
    This reduces the dimension of in1 to match out1. This is achieved by a
    summation of blocks of dimension (t_int, f_int).
    """

    cdef int d, t, f, aa, ab, rr, rc = 0
    cdef int n_dir, n_tim, n_fre, n_ant

    n_dir = m.shape[0]
    n_tim = m.shape[1]
    n_fre = m.shape[2]
    n_ant = m.shape[3]

    for d in xrange(n_dir):
        for t in xrange(n_tim):
            rr = t/t_int
            for f in xrange(n_fre):
                rc = f/f_int
                for aa in xrange(n_ant):
                    for ab in xrange(n_ant):
                        r[rr,rc,aa,ab,0,0] = \
                            r[rr,rc,aa,ab,0,0] + o[t,f,aa,ab,0,0] - (
                        g[d,rr,rc,aa,0,0]*m[d,t,f,aa,ab,0,0]*gh[d,rr,rc,ab,0,0] + \
                        g[d,rr,rc,aa,0,1]*m[d,t,f,aa,ab,1,0]*gh[d,rr,rc,ab,0,0] + \
                        g[d,rr,rc,aa,0,0]*m[d,t,f,aa,ab,0,1]*gh[d,rr,rc,ab,1,0] + \
                        g[d,rr,rc,aa,0,1]*m[d,t,f,aa,ab,1,1]*gh[d,rr,rc,ab,1,0])

                        r[rr,rc,aa,ab,0,1] = \
                            r[rr,rc,aa,ab,0,1] + o[t,f,aa,ab,0,1] - (
                        g[d,rr,rc,aa,0,0]*m[d,t,f,aa,ab,0,0]*gh[d,rr,rc,ab,0,1] + \
                        g[d,rr,rc,aa,0,1]*m[d,t,f,aa,ab,1,0]*gh[d,rr,rc,ab,0,1] + \
                        g[d,rr,rc,aa,0,0]*m[d,t,f,aa,ab,0,1]*gh[d,rr,rc,ab,1,1] + \
                        g[d,rr,rc,aa,0,1]*m[d,t,f,aa,ab,1,1]*gh[d,rr,rc,ab,1,1])

                        r[rr,rc,aa,ab,1,0] = \
                            r[rr,rc,aa,ab,1,0] + o[t,f,aa,ab,1,0] - (
                        g[d,rr,rc,aa,1,0]*m[d,t,f,aa,ab,0,0]*gh[d,rr,rc,ab,0,0] + \
                        g[d,rr,rc,aa,1,1]*m[d,t,f,aa,ab,1,0]*gh[d,rr,rc,ab,0,0] + \
                        g[d,rr,rc,aa,1,0]*m[d,t,f,aa,ab,0,1]*gh[d,rr,rc,ab,1,0] + \
                        g[d,rr,rc,aa,1,1]*m[d,t,f,aa,ab,1,1]*gh[d,rr,rc,ab,1,0])

                        r[rr,rc,aa,ab,1,1] = \
                            r[rr,rc,aa,ab,1,1] + o[t,f,aa,ab,1,1] - (
                        g[d,rr,rc,aa,1,0]*m[d,t,f,aa,ab,0,0]*gh[d,rr,rc,ab,0,1] + \
                        g[d,rr,rc,aa,1,1]*m[d,t,f,aa,ab,1,0]*gh[d,rr,rc,ab,0,1] + \
                        g[d,rr,rc,aa,1,0]*m[d,t,f,aa,ab,0,1]*gh[d,rr,rc,ab,1,1] + \
                        g[d,rr,rc,aa,1,1]*m[d,t,f,aa,ab,1,1]*gh[d,rr,rc,ab,1,1])

@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def cycompute_jh(complex3264 [:,:,:,:,:,:,:] m,
                 complex3264 [:,:,:,:,:,:] g,
                 complex3264 [:,:,:,:,:,:,:] jh,
                 int t_int,
                 int f_int):

    """
    This reduces the dimension of in1 to match out1. This is achieved by a
    summation of blocks of dimension (t_int, f_int).
    """

    cdef int d, t, f, aa, ab, rr, rc = 0
    cdef int n_dir, n_tim, n_fre, n_ant

    n_dir = m.shape[0]
    n_tim = m.shape[1]
    n_fre = m.shape[2]
    n_ant = m.shape[3]

    for d in xrange(n_dir):
        for t in xrange(n_tim):
            rr = t/t_int
            for f in xrange(n_fre):
                rc = f/f_int
                for aa in xrange(n_ant):
                    for ab in xrange(n_ant):
                        jh[d,t,f,aa,ab,0,0] = \
                                 g[d,rr,rc,aa,0,0]*m[d,t,f,aa,ab,0,0] + \
                                 g[d,rr,rc,aa,0,1]*m[d,t,f,aa,ab,1,0]

                        jh[d,t,f,aa,ab,0,1] = \
                                 g[d,rr,rc,aa,0,0]*m[d,t,f,aa,ab,0,1] + \
                                 g[d,rr,rc,aa,0,1]*m[d,t,f,aa,ab,1,1]

                        jh[d,t,f,aa,ab,1,0] = \
                                 g[d,rr,rc,aa,1,0]*m[d,t,f,aa,ab,0,0] + \
                                 g[d,rr,rc,aa,1,1]*m[d,t,f,aa,ab,1,0]

                        jh[d,t,f,aa,ab,1,1] = \
                                 g[d,rr,rc,aa,1,0]*m[d,t,f,aa,ab,0,1] + \
                                 g[d,rr,rc,aa,1,1]*m[d,t,f,aa,ab,1,1]

@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def cycompute_jhr(complex3264 [:,:,:,:,:,:,:] jh,
                  complex3264 [:,:,:,:,:,:] r,
                  complex3264 [:,:,:,:,:,:] jhr,
                  int t_int,
                  int f_int):

    """
    This reduces the dimension of in1 to match out1. This is achieved by a
    summation of blocks of dimension (t_int, f_int).
    """

    cdef int d, t, f, aa, ab, rr, rc = 0
    cdef int n_dir, n_tim, n_fre, n_ant

    n_dir = jh.shape[0]
    n_tim = jh.shape[1]
    n_fre = jh.shape[2]
    n_ant = jh.shape[3]

    for d in xrange(n_dir):
        for t in xrange(n_tim):
            rr = t/t_int
            for f in xrange(n_fre):
                rc = f/f_int
                for aa in xrange(n_ant):
                    for ab in xrange(n_ant):
                        jhr[d,rr,rc,aa,0,0] = jhr[d,rr,rc,aa,0,0] + \
                             r[t,f,aa,ab,0,0]*jh[d,t,f,ab,aa,0,0] + \
                             r[t,f,aa,ab,0,1]*jh[d,t,f,ab,aa,1,0]

                        jhr[d,rr,rc,aa,0,1] = jhr[d,rr,rc,aa,0,1] + \
                             r[t,f,aa,ab,0,0]*jh[d,t,f,ab,aa,0,1] + \
                             r[t,f,aa,ab,0,1]*jh[d,t,f,ab,aa,1,1]

                        jhr[d,rr,rc,aa,1,0] = jhr[d,rr,rc,aa,1,0] + \
                             r[t,f,aa,ab,1,0]*jh[d,t,f,ab,aa,0,0] + \
                             r[t,f,aa,ab,1,1]*jh[d,t,f,ab,aa,1,0]

                        jhr[d,rr,rc,aa,1,1] = jhr[d,rr,rc,aa,1,1] + \
                             r[t,f,aa,ab,1,0]*jh[d,t,f,ab,aa,0,1] + \
                             r[t,f,aa,ab,1,1]*jh[d,t,f,ab,aa,1,1]

@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def cycompute_jhj(complex3264 [:,:,:,:,:,:,:] jh,
                  complex3264 [:,:,:,:,:,:] jhj,
                  int t_int,
                  int f_int):

    """
    This reduces the dimension of in1 to match out1. This is achieved by a
    summation of blocks of dimension (t_int, f_int).
    """

    cdef int d, t, f, aa, ab, rr, rc = 0
    cdef int n_dir, n_tim, n_fre, n_ant

    n_dir = jh.shape[0]
    n_tim = jh.shape[1]
    n_fre = jh.shape[2]
    n_ant = jh.shape[3]

    for d in xrange(n_dir):
        for t in xrange(n_tim):
            rr = t/t_int
            for f in xrange(n_fre):
                rc = f/f_int
                for aa in xrange(n_ant):
                    for ab in xrange(n_ant):
                        jhj[d,rr,rc,aa,0,0] = jhj[d,rr,rc,aa,0,0] + \
                        jh[d,t,f,ab,aa,0,0].conjugate()*jh[d,t,f,ab,aa,0,0] + \
                        jh[d,t,f,ab,aa,1,0].conjugate()*jh[d,t,f,ab,aa,1,0]

                        jhj[d,rr,rc,aa,0,1] = jhj[d,rr,rc,aa,0,1] + \
                        jh[d,t,f,ab,aa,0,0].conjugate()*jh[d,t,f,ab,aa,0,1] + \
                        jh[d,t,f,ab,aa,1,0].conjugate()*jh[d,t,f,ab,aa,1,1]

                        jhj[d,rr,rc,aa,1,0] = jhj[d,rr,rc,aa,1,0] + \
                        jh[d,t,f,ab,aa,0,1].conjugate()*jh[d,t,f,ab,aa,0,0] + \
                        jh[d,t,f,ab,aa,1,1].conjugate()*jh[d,t,f,ab,aa,1,0]

                        jhj[d,rr,rc,aa,1,1] = jhj[d,rr,rc,aa,1,1] + \
                        jh[d,t,f,ab,aa,0,1].conjugate()*jh[d,t,f,ab,aa,0,1] + \
                        jh[d,t,f,ab,aa,1,1].conjugate()*jh[d,t,f,ab,aa,1,1]

@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def cycompute_jhjinv(complex3264 [:,:,:,:,:,:] jhj,
                     complex3264 [:,:,:,:,:,:] jhjinv):

    cdef int d, t, f, aa, ab = 0
    cdef int n_dir, n_tim, n_fre, n_ant
    cdef complex3264 denom = 0

    n_dir = jhj.shape[0]
    n_tim = jhj.shape[1]
    n_fre = jhj.shape[2]
    n_ant = jhj.shape[3]

    for d in xrange(n_dir):
        for t in xrange(n_tim):
            for f in xrange(n_fre):
                for aa in xrange(n_ant):

                    denom = jhj[d,t,f,aa,0,0] * jhj[d,t,f,aa,1,1] - \
                            jhj[d,t,f,aa,0,1] * jhj[d,t,f,aa,1,0]

                    if denom==0:
                        denom = 1

                    jhjinv[d,t,f,aa,0,0] = jhj[d,t,f,aa,1,1]/denom
                    jhjinv[d,t,f,aa,1,1] = jhj[d,t,f,aa,0,0]/denom
                    jhjinv[d,t,f,aa,0,1] = -1 * jhj[d,t,f,aa,0,1]/denom
                    jhjinv[d,t,f,aa,1,0] = -1 * jhj[d,t,f,aa,1,0]/denom


@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def cycompute_update(complex3264 [:,:,:,:,:,:] jhr,
                     complex3264 [:,:,:,:,:,:] jhj,
                     complex3264 [:,:,:,:,:,:] upd):
    """
    NOTE: THIS RIGHT-MULTIPLIES THE ENTRIES OF IN1 BY THE ENTRIES OF IN2.
    """

    cdef int d, t, f, aa = 0
    cdef int n_dir, n_tim, n_fre, n_ant

    n_dir = jhr.shape[0]
    n_tim = jhr.shape[1]
    n_fre = jhr.shape[2]
    n_ant = jhr.shape[3]

    for d in xrange(n_dir):
        for t in xrange(n_tim):
            for f in xrange(n_fre):
                for aa in xrange(n_ant):

                    upd[d,t,f,aa,0,0] = jhr[d,t,f,aa,0,0]*jhj[d,t,f,aa,0,0] + \
                                        jhr[d,t,f,aa,0,1]*jhj[d,t,f,aa,1,0]

                    upd[d,t,f,aa,0,1] = jhr[d,t,f,aa,0,0]*jhj[d,t,f,aa,0,1] + \
                                        jhr[d,t,f,aa,0,1]*jhj[d,t,f,aa,1,1]

                    upd[d,t,f,aa,1,0] = jhr[d,t,f,aa,1,0]*jhj[d,t,f,aa,0,0] + \
                                        jhr[d,t,f,aa,1,1]*jhj[d,t,f,aa,1,0]

                    upd[d,t,f,aa,1,1] = jhr[d,t,f,aa,1,0]*jhj[d,t,f,aa,0,1] + \
                                        jhr[d,t,f,aa,1,1]*jhj[d,t,f,aa,1,1]
