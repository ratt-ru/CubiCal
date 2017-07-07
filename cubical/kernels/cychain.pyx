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
def cyapply_left_inv_jones(complex3264 [:,:,:,:,:,:] jhr,
                           complex3264 [:,:,:,:,:,:] ginv,
                           int t_int,
                           int f_int):

    """
    This will apply the jones term to the left side of jhr.
    """

    cdef int d, i, t, f, aa, ab, rr, rc = 0
    cdef int n_dir, n_tim, n_fre, n_ant
    cdef complex3264 jhr00, jhr01, jhr10, jhr11

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

                    jhr00 = jhr[d,t,f,aa,0,0]
                    jhr01 = jhr[d,t,f,aa,0,1]
                    jhr10 = jhr[d,t,f,aa,1,0]
                    jhr11 = jhr[d,t,f,aa,1,1]

                    jhr[d,t,f,aa,0,0] = ginv[d,rr,rc,aa,0,0]*jhr00 + \
                                        ginv[d,rr,rc,aa,0,1]*jhr10

                    jhr[d,t,f,aa,0,1] = ginv[d,rr,rc,aa,0,0]*jhr01 + \
                                        ginv[d,rr,rc,aa,0,1]*jhr11

                    jhr[d,t,f,aa,1,0] = ginv[d,rr,rc,aa,1,0]*jhr00 + \
                                        ginv[d,rr,rc,aa,1,1]*jhr10

                    jhr[d,t,f,aa,1,1] = ginv[d,rr,rc,aa,1,0]*jhr01 + \
                                        ginv[d,rr,rc,aa,1,1]*jhr11

@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def cysum_jhr_intervals(complex3264 [:,:,:,:,:,:] jhr,
                        complex3264 [:,:,:,:,:,:] jhrint,
                        int t_int,
                        int f_int):

    """
    This will sum over the solution interval of the current term.
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
    This computes the residual, resulting in large matrix indexed by 
    (direction, model, time, frequency, antenna, antenna, correclation, correlation).
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