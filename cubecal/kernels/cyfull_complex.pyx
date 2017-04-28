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
def cycompute_residual(np.ndarray[complex3264, ndim=8] m,
                       np.ndarray[complex3264, ndim=6] g,
                       np.ndarray[complex3264, ndim=6] gh,
                       np.ndarray[complex3264, ndim=7] o,
                       np.ndarray[complex3264, ndim=7] r,
                       int t_int,
                       int f_int):

    """
    This computes the residual, resulting in large matrix indexed by 
    (direction, model, time, frequency, antenna, antenna, correclation, correlation).
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
                            r[i,t,f,aa,ab,0,0] = o[i,t,f,aa,ab,0,0] - (
                            g[d,rr,rc,aa,0,0]*m[d,i,t,f,aa,ab,0,0]*gh[d,rr,rc,ab,0,0] + \
                            g[d,rr,rc,aa,0,1]*m[d,i,t,f,aa,ab,1,0]*gh[d,rr,rc,ab,0,0] + \
                            g[d,rr,rc,aa,0,0]*m[d,i,t,f,aa,ab,0,1]*gh[d,rr,rc,ab,1,0] + \
                            g[d,rr,rc,aa,0,1]*m[d,i,t,f,aa,ab,1,1]*gh[d,rr,rc,ab,1,0])

                            r[i,t,f,aa,ab,0,1] = o[i,t,f,aa,ab,0,1] - (
                            g[d,rr,rc,aa,0,0]*m[d,i,t,f,aa,ab,0,0]*gh[d,rr,rc,ab,0,1] + \
                            g[d,rr,rc,aa,0,1]*m[d,i,t,f,aa,ab,1,0]*gh[d,rr,rc,ab,0,1] + \
                            g[d,rr,rc,aa,0,0]*m[d,i,t,f,aa,ab,0,1]*gh[d,rr,rc,ab,1,1] + \
                            g[d,rr,rc,aa,0,1]*m[d,i,t,f,aa,ab,1,1]*gh[d,rr,rc,ab,1,1])

                            r[i,t,f,aa,ab,1,0] = o[i,t,f,aa,ab,1,0] - (
                            g[d,rr,rc,aa,1,0]*m[d,i,t,f,aa,ab,0,0]*gh[d,rr,rc,ab,0,0] + \
                            g[d,rr,rc,aa,1,1]*m[d,i,t,f,aa,ab,1,0]*gh[d,rr,rc,ab,0,0] + \
                            g[d,rr,rc,aa,1,0]*m[d,i,t,f,aa,ab,0,1]*gh[d,rr,rc,ab,1,0] + \
                            g[d,rr,rc,aa,1,1]*m[d,i,t,f,aa,ab,1,1]*gh[d,rr,rc,ab,1,0])

                            r[i,t,f,aa,ab,1,1] = o[i,t,f,aa,ab,1,1] - (
                            g[d,rr,rc,aa,1,0]*m[d,i,t,f,aa,ab,0,0]*gh[d,rr,rc,ab,0,1] + \
                            g[d,rr,rc,aa,1,1]*m[d,i,t,f,aa,ab,1,0]*gh[d,rr,rc,ab,0,1] + \
                            g[d,rr,rc,aa,1,0]*m[d,i,t,f,aa,ab,0,1]*gh[d,rr,rc,ab,1,1] + \
                            g[d,rr,rc,aa,1,1]*m[d,i,t,f,aa,ab,1,1]*gh[d,rr,rc,ab,1,1])

@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def cycompute_jh(np.ndarray[complex3264, ndim=8] m,
                 np.ndarray[complex3264, ndim=6] g,
                 np.ndarray[complex3264, ndim=8] jh,
                 int t_int,
                 int f_int):

    """
    This computes the the non-zero elements of jh. The result does have a model index.  
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
                            jh[d,i,t,f,aa,ab,0,0] = g[d,rr,rc,aa,0,0]*m[d,i,t,f,aa,ab,0,0] + \
                                                    g[d,rr,rc,aa,0,1]*m[d,i,t,f,aa,ab,1,0]

                            jh[d,i,t,f,aa,ab,0,1] = g[d,rr,rc,aa,0,0]*m[d,i,t,f,aa,ab,0,1] + \
                                                    g[d,rr,rc,aa,0,1]*m[d,i,t,f,aa,ab,1,1]

                            jh[d,i,t,f,aa,ab,1,0] = g[d,rr,rc,aa,1,0]*m[d,i,t,f,aa,ab,0,0] + \
                                                    g[d,rr,rc,aa,1,1]*m[d,i,t,f,aa,ab,1,0]

                            jh[d,i,t,f,aa,ab,1,1] = g[d,rr,rc,aa,1,0]*m[d,i,t,f,aa,ab,0,1] + \
                                                    g[d,rr,rc,aa,1,1]*m[d,i,t,f,aa,ab,1,1]

@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def cycompute_jhr(np.ndarray[complex3264, ndim=8] jh,
                  np.ndarray[complex3264, ndim=7] r,
                  np.ndarray[complex3264, ndim=6] jhr,
                  int t_int,
                  int f_int):

    """
    This computes the jhr term on the GN/LM method. Note that while jh is indexed by model, the 
    resulting jhr has no model index. 
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
                            jhr[d,rr,rc,aa,0,0] = jhr[d,rr,rc,aa,0,0] + \
                                                    r[i,t,f,aa,ab,0,0]*jh[d,i,t,f,ab,aa,0,0] + \
                                                    r[i,t,f,aa,ab,0,1]*jh[d,i,t,f,ab,aa,1,0]

                            jhr[d,rr,rc,aa,0,1] = jhr[d,rr,rc,aa,0,1] + \
                                                    r[i,t,f,aa,ab,0,0]*jh[d,i,t,f,ab,aa,0,1] + \
                                                    r[i,t,f,aa,ab,0,1]*jh[d,i,t,f,ab,aa,1,1]

                            jhr[d,rr,rc,aa,1,0] = jhr[d,rr,rc,aa,1,0] + \
                                                    r[i,t,f,aa,ab,1,0]*jh[d,i,t,f,ab,aa,0,0] + \
                                                    r[i,t,f,aa,ab,1,1]*jh[d,i,t,f,ab,aa,1,0]

                            jhr[d,rr,rc,aa,1,1] = jhr[d,rr,rc,aa,1,1] + \
                                                    r[i,t,f,aa,ab,1,0]*jh[d,i,t,f,ab,aa,0,1] + \
                                                    r[i,t,f,aa,ab,1,1]*jh[d,i,t,f,ab,aa,1,1]

@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def cycompute_jhj(np.ndarray[complex3264, ndim=8] jh,
                  np.ndarray[complex3264, ndim=6] jhj,
                  int t_int,
                  int f_int):

    """
    This computes the approximation to the Hessian, jhj. 
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
                            jhj[d,rr,rc,aa,0,0] = jhj[d,rr,rc,aa,0,0] + \
                            jh[d,i,t,f,ab,aa,0,0].conjugate()*jh[d,i,t,f,ab,aa,0,0] + \
                            jh[d,i,t,f,ab,aa,1,0].conjugate()*jh[d,i,t,f,ab,aa,1,0]

                            jhj[d,rr,rc,aa,0,1] = jhj[d,rr,rc,aa,0,1] + \
                            jh[d,i,t,f,ab,aa,0,0].conjugate()*jh[d,i,t,f,ab,aa,0,1] + \
                            jh[d,i,t,f,ab,aa,1,0].conjugate()*jh[d,i,t,f,ab,aa,1,1]

                            jhj[d,rr,rc,aa,1,0] = jhj[d,rr,rc,aa,1,0] + \
                            jh[d,i,t,f,ab,aa,0,1].conjugate()*jh[d,i,t,f,ab,aa,0,0] + \
                            jh[d,i,t,f,ab,aa,1,1].conjugate()*jh[d,i,t,f,ab,aa,1,0]

                            jhj[d,rr,rc,aa,1,1] = jhj[d,rr,rc,aa,1,1] + \
                            jh[d,i,t,f,ab,aa,0,1].conjugate()*jh[d,i,t,f,ab,aa,0,1] + \
                            jh[d,i,t,f,ab,aa,1,1].conjugate()*jh[d,i,t,f,ab,aa,1,1]

@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def cycompute_jhjinv(np.ndarray[complex3264, ndim=6] jhj,
                     np.ndarray[complex3264, ndim=6] jhjinv):
    """
    This inverts the approximation to the Hessian, jhj. Note that asa as useful side effect, it is 
    also suitable for inverting the gains.
    """

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
def cycompute_update(np.ndarray[complex3264, ndim=6] jhr,
                     np.ndarray[complex3264, ndim=6] jhj,
                     np.ndarray[complex3264, ndim=6] upd):
    """
    This computes the update by computing the product of jhj and jhr. These should already have been
    reduced to the correct dimension so that this operation is very simple. 
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
                    

@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def cycompute_corrected(np.ndarray[complex3264, ndim=6] o,
                        np.ndarray[complex3264, ndim=6] g,
                        np.ndarray[complex3264, ndim=6] gh,
                        np.ndarray[complex3264, ndim=6] corr,
                        int t_int,
                        int f_int):

    """
    This computes the corrected visiblities, given the observed visiblities and the inverse of the 
    gains. Note that the observed array expected here MUST NOT have a model index. This is because 
    we only obtain a single gain estimate across all models and want to apply it to the raw the 
    visibility data.
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
                        g[d,rr,rc,aa,0,0]*o[t,f,aa,ab,0,0]*gh[d,rr,rc,ab,0,0] + \
                        g[d,rr,rc,aa,0,1]*o[t,f,aa,ab,1,0]*gh[d,rr,rc,ab,0,0] + \
                        g[d,rr,rc,aa,0,0]*o[t,f,aa,ab,0,1]*gh[d,rr,rc,ab,1,0] + \
                        g[d,rr,rc,aa,0,1]*o[t,f,aa,ab,1,1]*gh[d,rr,rc,ab,1,0]

                        corr[t,f,aa,ab,0,1] = \
                        g[d,rr,rc,aa,0,0]*o[t,f,aa,ab,0,0]*gh[d,rr,rc,ab,0,1] + \
                        g[d,rr,rc,aa,0,1]*o[t,f,aa,ab,1,0]*gh[d,rr,rc,ab,0,1] + \
                        g[d,rr,rc,aa,0,0]*o[t,f,aa,ab,0,1]*gh[d,rr,rc,ab,1,1] + \
                        g[d,rr,rc,aa,0,1]*o[t,f,aa,ab,1,1]*gh[d,rr,rc,ab,1,1]

                        corr[t,f,aa,ab,1,0] = \
                        g[d,rr,rc,aa,1,0]*o[t,f,aa,ab,0,0]*gh[d,rr,rc,ab,0,0] + \
                        g[d,rr,rc,aa,1,1]*o[t,f,aa,ab,1,0]*gh[d,rr,rc,ab,0,0] + \
                        g[d,rr,rc,aa,1,0]*o[t,f,aa,ab,0,1]*gh[d,rr,rc,ab,1,0] + \
                        g[d,rr,rc,aa,1,1]*o[t,f,aa,ab,1,1]*gh[d,rr,rc,ab,1,0]

                        corr[t,f,aa,ab,1,1] = \
                        g[d,rr,rc,aa,1,0]*o[t,f,aa,ab,0,0]*gh[d,rr,rc,ab,0,1] + \
                        g[d,rr,rc,aa,1,1]*o[t,f,aa,ab,1,0]*gh[d,rr,rc,ab,0,1] + \
                        g[d,rr,rc,aa,1,0]*o[t,f,aa,ab,0,1]*gh[d,rr,rc,ab,1,1] + \
                        g[d,rr,rc,aa,1,1]*o[t,f,aa,ab,1,1]*gh[d,rr,rc,ab,1,1]
