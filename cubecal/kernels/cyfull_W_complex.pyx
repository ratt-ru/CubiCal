from cython.parallel import prange, parallel
import numpy as np
cimport numpy as np
import cython
from scipy.optimize import fsolve
from scipy import special

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
    (direction, model, time, frequency, antenna, antenna, correlation, correlation).
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
def cycompute_jhwr(np.ndarray[complex3264, ndim=8] jh,
                  np.ndarray[complex3264, ndim=7] r,
                  np.ndarray[np.double_t, ndim=5] w,
                  np.ndarray[complex3264, ndim=6] jhwr,
                  int t_int,
                  int f_int):

    """
    This computes the jhwr term on the GN/LM method. Note that while jh is indexed by model, the 
    resulting jhwr has no model index. 
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
                            jhwr[d,rr,rc,aa,0,0] = jhwr[d,rr,rc,aa,0,0] + \
                                                    w[i,t,f,aa,ab]*r[i,t,f,aa,ab,0,0]*jh[d,i,t,f,ab,aa,0,0] + \
                                                    w[i,t,f,aa,ab]*r[i,t,f,aa,ab,0,1]*jh[d,i,t,f,ab,aa,1,0]

                            jhwr[d,rr,rc,aa,0,1] = jhwr[d,rr,rc,aa,0,1] + \
                                                    w[i,t,f,aa,ab]*r[i,t,f,aa,ab,0,0]*jh[d,i,t,f,ab,aa,0,1] + \
                                                    w[i,t,aa,ab]*r[i,t,f,aa,ab,0,1]*jh[d,i,t,f,ab,aa,1,1]

                            jhwr[d,rr,rc,aa,1,0] = jhwr[d,rr,rc,aa,1,0] + \
                                                    w[i,t,f,aa,ab]*r[i,t,f,aa,ab,1,0]*jh[d,i,t,f,ab,aa,0,0] + \
                                                    w[i,t,f,aa,ab]*r[i,t,f,aa,ab,1,1]*jh[d,i,t,f,ab,aa,1,0]

                            jhwr[d,rr,rc,aa,1,1] = jhwr[d,rr,rc,aa,1,1] + \
                                                    w[i,t,f,aa,ab]*r[i,t,f,aa,ab,1,0]*jh[d,i,t,f,ab,aa,0,1] + \
                                                    w[i,t,f,aa,ab]*r[i,t,f,aa,ab,1,1]*jh[d,i,t,f,ab,aa,1,1]

@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def cycompute_jhwj(np.ndarray[complex3264, ndim=8] jh,
                  np.ndarray[np.double_t, ndim=5] w,  
                  np.ndarray[complex3264, ndim=6] jhwj,
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
                            jhwj[d,rr,rc,aa,0,0] = jhwj[d,rr,rc,aa,0,0] + \
                            jh[d,i,t,f,ab,aa,0,0].conjugate()*w[i,t,f,aa,ab]*jh[d,i,t,f,ab,aa,0,0] + \
                            jh[d,i,t,f,ab,aa,1,0].conjugate()*w[i,t,f,aa,ab]*jh[d,i,t,f,ab,aa,1,0]

                            jhwj[d,rr,rc,aa,0,1] = jhwj[d,rr,rc,aa,0,1] + \
                            jh[d,i,t,f,ab,aa,0,0].conjugate()*w[i,t,f,aa,ab]*jh[d,i,t,f,ab,aa,0,1] + \
                            jh[d,i,t,f,ab,aa,1,0].conjugate()*w[i,t,f,aa,ab]*jh[d,i,t,f,ab,aa,1,1]

                            jhwj[d,rr,rc,aa,1,0] = jhwj[d,rr,rc,aa,1,0] + \
                            jh[d,i,t,f,ab,aa,0,1].conjugate()*w[i,t,f,aa,ab]*jh[d,i,t,f,ab,aa,0,0] + \
                            jh[d,i,t,f,ab,aa,1,1].conjugate()*w[i,t,f,aa,ab]*jh[d,i,t,f,ab,aa,1,0]

                            jhwj[d,rr,rc,aa,1,1] = jhwj[d,rr,rc,aa,1,1] + \
                            jh[d,i,t,f,ab,aa,0,1].conjugate()*w[i,t,f,aa,ab]*jh[d,i,t,f,ab,aa,0,1] + \
                            jh[d,i,t,f,ab,aa,1,1].conjugate()*w[i,t,f,aa,ab]*jh[d,i,t,f,ab,aa,1,1]

@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def cycompute_jhwjinv(np.ndarray[complex3264, ndim=6] jhwj,
                     np.ndarray[complex3264, ndim=6] jhwjinv):
    """
    This inverts the approximation to the Hessian, jhwj. Note that as a useful side effect, it is 
    also suitable for inverting the gains.
    """


    cdef int d, t, f, aa, ab = 0
    cdef int n_dir, n_tim, n_fre, n_ant
    cdef complex3264 denom = 0

    n_dir = jhwj.shape[0]
    n_tim = jhwj.shape[1]
    n_fre = jhwj.shape[2]
    n_ant = jhwj.shape[3]

    for d in xrange(n_dir):
        for t in xrange(n_tim):
            for f in xrange(n_fre):
                for aa in xrange(n_ant):

                    denom = jhwj[d,t,f,aa,0,0] * jhwj[d,t,f,aa,1,1] - \
                            jhwj[d,t,f,aa,0,1] * jhwj[d,t,f,aa,1,0]

                    if denom==0:
                        denom = 1

                    jhwjinv[d,t,f,aa,0,0] = jhwj[d,t,f,aa,1,1]/denom
                    jhwjinv[d,t,f,aa,1,1] = jhwj[d,t,f,aa,0,0]/denom
                    jhwjinv[d,t,f,aa,0,1] = -1 * jhwj[d,t,f,aa,0,1]/denom
                    jhwjinv[d,t,f,aa,1,0] = -1 * jhwj[d,t,f,aa,1,0]/denom


@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def cycompute_update(np.ndarray[complex3264, ndim=6] jhwr,
                     np.ndarray[complex3264, ndim=6] jhwj,
                     np.ndarray[complex3264, ndim=6] upd):
    """
    This computes the update by computing the product of jhwj and jhwr. These should already have been
    reduced to the correct dimension so that this operation is very simple. 
    """

    cdef int d, t, f, aa = 0
    cdef int n_dir, n_tim, n_fre, n_ant

    n_dir = jhwr.shape[0]
    n_tim = jhwr.shape[1]
    n_fre = jhwr.shape[2]
    n_ant = jhwr.shape[3]

    for d in xrange(n_dir):
        for t in xrange(n_tim):
            for f in xrange(n_fre):
                for aa in xrange(n_ant):

                    upd[d,t,f,aa,0,0] = jhwr[d,t,f,aa,0,0]*jhwj[d,t,f,aa,0,0] + \
                                        jhwr[d,t,f,aa,0,1]*jhwj[d,t,f,aa,1,0]

                    upd[d,t,f,aa,0,1] = jhwr[d,t,f,aa,0,0]*jhwj[d,t,f,aa,0,1] + \
                                        jhwr[d,t,f,aa,0,1]*jhwj[d,t,f,aa,1,1]

                    upd[d,t,f,aa,1,0] = jhwr[d,t,f,aa,1,0]*jhwj[d,t,f,aa,0,0] + \
                                        jhwr[d,t,f,aa,1,1]*jhwj[d,t,f,aa,1,0]

                    upd[d,t,f,aa,1,1] = jhwr[d,t,f,aa,1,0]*jhwj[d,t,f,aa,0,1] + \
                                        jhwr[d,t,f,aa,1,1]*jhwj[d,t,f,aa,1,1]
                    

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


@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def cycompute_weights(np.ndarray[complex3264, ndim=7] r,
                        np.ndarray[np.double_t, ndim=5] w,
                        np.ndarray[np.double_t, ndim=3] v,
                        int t_int,
                        int f_int):
    """
    This computes the weights, given the latest residual visibilities and the v parameter.
    w[i] = (v+2)/(v + 2*residual[i]^2). Next v is update using the newly compute weights.
    """

    cdef i, t, f, aa, ab, rr, rc = 0
    cdef int n_mod, n_tim, n_fre, n_ant, m
    cdef double d
    cdef np.ndarray[np.double_t, ndim=1] winit
    cdef np.double_t[:] wn

    n_mod = r.shape[0]
    n_tim = r.shape[1]
    n_fre = r.shape[2]
    n_ant = r.shape[3]

    for i in xrange(n_mod):
        for t in xrange(n_tim):
            rr = t/t_int
            for f in xrange(n_fre):
                rc = f/f_int
                for aa in xrange(n_ant):
                    for ab in xrange(n_ant):
                        w[i,t,f,aa,ab] = (v[i,rr,rc]+2)/(v[i,rr,rc] + \
                                2*(r[i,t,f,aa,ab,0,0].conjugate()*r[i,t,f,aa,ab,0,0] + \
                                r[i,t,f,aa,ab,0,1].conjugate()*r[i,t,f,aa,ab,0,1] + \
                                r[i,t,f,aa,ab,1,0].conjugate()*r[i,t,f,aa,ab,1,0] + \
                                r[i,t,f,aa,ab,1,1].conjugate()*r[i,t,f,aa,ab,1,1]))

    rr = n_tim/t_int
    rc = n_fre/f_int

    for i in xrange(n_mod):
        for t in xrange(0, n_tim, t_int):
            for f in xrange(0, n_fre, f_int):
                winit = np.reshape(w[t:t+t_int,f+f_int,:,:],(t_int*f_int*n_ant*n_ant))
                wn = winit[np.where(winit!=0)]
                m = len(wn)

                vfunc = lambda a: special.digamma(0.5*(a+2)) - np.log(0.5*(a+2)) - special.digamma(0.5*a) + np.log(0.5*a) + (1./m)*np.sum(np.log(wn) - wn) + 1


                d = fsolve(vfunc,v[i,t,f])
                if d > 30 or d<2:
                    v[i,t,f] = v[i,t,f]
                else:
                    v[i,t,f] = d