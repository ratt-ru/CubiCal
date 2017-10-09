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
def cycompute_residual(complex3264 [:,:,:,:,:,:,:,:] m,
                       complex3264 [:,:,:,:,:,:] g,
                       complex3264 [:,:,:,:,:,:] gh,
                       complex3264 [:,:,:,:,:,:,:] r,
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
                            r[i,t,f,aa,ab,0,0] = r[i,t,f,aa,ab,0,0] - (
                            g[d,rr,rc,aa,0,0]*m[d,i,t,f,aa,ab,0,0]*gh[d,rr,rc,ab,0,0] + \
                            g[d,rr,rc,aa,0,1]*m[d,i,t,f,aa,ab,1,0]*gh[d,rr,rc,ab,0,0] + \
                            g[d,rr,rc,aa,0,0]*m[d,i,t,f,aa,ab,0,1]*gh[d,rr,rc,ab,1,0] + \
                            g[d,rr,rc,aa,0,1]*m[d,i,t,f,aa,ab,1,1]*gh[d,rr,rc,ab,1,0])

                            r[i,t,f,aa,ab,0,1] = r[i,t,f,aa,ab,0,1] - (
                            g[d,rr,rc,aa,0,0]*m[d,i,t,f,aa,ab,0,0]*gh[d,rr,rc,ab,0,1] + \
                            g[d,rr,rc,aa,0,1]*m[d,i,t,f,aa,ab,1,0]*gh[d,rr,rc,ab,0,1] + \
                            g[d,rr,rc,aa,0,0]*m[d,i,t,f,aa,ab,0,1]*gh[d,rr,rc,ab,1,1] + \
                            g[d,rr,rc,aa,0,1]*m[d,i,t,f,aa,ab,1,1]*gh[d,rr,rc,ab,1,1])

                            r[i,t,f,aa,ab,1,0] = r[i,t,f,aa,ab,1,0] - (
                            g[d,rr,rc,aa,1,0]*m[d,i,t,f,aa,ab,0,0]*gh[d,rr,rc,ab,0,0] + \
                            g[d,rr,rc,aa,1,1]*m[d,i,t,f,aa,ab,1,0]*gh[d,rr,rc,ab,0,0] + \
                            g[d,rr,rc,aa,1,0]*m[d,i,t,f,aa,ab,0,1]*gh[d,rr,rc,ab,1,0] + \
                            g[d,rr,rc,aa,1,1]*m[d,i,t,f,aa,ab,1,1]*gh[d,rr,rc,ab,1,0])

                            r[i,t,f,aa,ab,1,1] = r[i,t,f,aa,ab,1,1] - (
                            g[d,rr,rc,aa,1,0]*m[d,i,t,f,aa,ab,0,0]*gh[d,rr,rc,ab,0,1] + \
                            g[d,rr,rc,aa,1,1]*m[d,i,t,f,aa,ab,1,0]*gh[d,rr,rc,ab,0,1] + \
                            g[d,rr,rc,aa,1,0]*m[d,i,t,f,aa,ab,0,1]*gh[d,rr,rc,ab,1,1] + \
                            g[d,rr,rc,aa,1,1]*m[d,i,t,f,aa,ab,1,1]*gh[d,rr,rc,ab,1,1])


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
def cycompute_jhwr(complex3264 [:,:,:,:,:,:,:,:] jh,
                  complex3264 [:,:,:,:,:,:,:] r,
                  complex3264 [:,:,:,:,:,:] w,
                  complex3264 [:,:,:,:,:,:] jhwr,
                  int t_int,
                  int f_int):

    """
    This computes the jhwr term on the weigted GN/LM method. Note that while jh is indexed by model, the 
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
                                                    w[i,t,f,aa,ab,0]*r[i,t,f,aa,ab,0,0]*jh[d,i,t,f,ab,aa,0,0] + \
                                                    w[i,t,f,aa,ab,0]*r[i,t,f,aa,ab,0,1]*jh[d,i,t,f,ab,aa,1,0]

                            jhwr[d,rr,rc,aa,0,1] = jhwr[d,rr,rc,aa,0,1] + \
                                                    w[i,t,f,aa,ab,0]*r[i,t,f,aa,ab,0,0]*jh[d,i,t,f,ab,aa,0,1] + \
                                                    w[i,t,f,aa,ab,0]*r[i,t,f,aa,ab,0,1]*jh[d,i,t,f,ab,aa,1,1]

                            jhwr[d,rr,rc,aa,1,0] = jhwr[d,rr,rc,aa,1,0] + \
                                                    w[i,t,f,aa,ab,0]*r[i,t,f,aa,ab,1,0]*jh[d,i,t,f,ab,aa,0,0] + \
                                                    w[i,t,f,aa,ab,0]*r[i,t,f,aa,ab,1,1]*jh[d,i,t,f,ab,aa,1,0]

                            jhwr[d,rr,rc,aa,1,1] = jhwr[d,rr,rc,aa,1,1] + \
                                                    w[i,t,f,aa,ab,0]*r[i,t,f,aa,ab,1,0]*jh[d,i,t,f,ab,aa,0,1] + \
                                                    w[i,t,f,aa,ab,0]*r[i,t,f,aa,ab,1,1]*jh[d,i,t,f,ab,aa,1,1]

@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def cycompute_jhwj(complex3264[:,:,:,:,:,:,:,:] jh,
                  complex3264[:,:,:,:,:,:] w,  
                  complex3264[:,:,:,:,:,:]jhwj,
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
                            jh[d,i,t,f,ab,aa,0,0].conjugate()*w[i,t,f,aa,ab,0]*jh[d,i,t,f,ab,aa,0,0] + \
                            jh[d,i,t,f,ab,aa,1,0].conjugate()*w[i,t,f,aa,ab,0]*jh[d,i,t,f,ab,aa,1,0]

                            jhwj[d,rr,rc,aa,0,1] = jhwj[d,rr,rc,aa,0,1] + \
                            jh[d,i,t,f,ab,aa,0,0].conjugate()*w[i,t,f,aa,ab,0]*jh[d,i,t,f,ab,aa,0,1] + \
                            jh[d,i,t,f,ab,aa,1,0].conjugate()*w[i,t,f,aa,ab,0]*jh[d,i,t,f,ab,aa,1,1]

                            jhwj[d,rr,rc,aa,1,0] = jhwj[d,rr,rc,aa,1,0] + \
                            jh[d,i,t,f,ab,aa,0,1].conjugate()*w[i,t,f,aa,ab,0]*jh[d,i,t,f,ab,aa,0,0] + \
                            jh[d,i,t,f,ab,aa,1,1].conjugate()*w[i,t,f,aa,ab,0]*jh[d,i,t,f,ab,aa,1,0]

                            jhwj[d,rr,rc,aa,1,1] = jhwj[d,rr,rc,aa,1,1] + \
                            jh[d,i,t,f,ab,aa,0,1].conjugate()*w[i,t,f,aa,ab,0]*jh[d,i,t,f,ab,aa,0,1] + \
                            jh[d,i,t,f,ab,aa,1,1].conjugate()*w[i,t,f,aa,ab,0]*jh[d,i,t,f,ab,aa,1,1]


@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def cycompute_jhwjinv(complex3264 [:,:,:,:,:,:] jhj,
                     complex3264 [:,:,:,:,:,:] jhjinv,
                     np.uint16_t [:,:,:,:] flags,
                     float eps,
                     int flagbit):
    """
    This inverts the approximation to the Hessian, jhj. Note that as a useful side effect, it is 
    also suitable for inverting the gains.

    Returns number of points flagged
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

    for d in xrange(n_dir):
        for t in xrange(n_tim):
            for f in xrange(n_fre):
                for aa in xrange(n_ant):
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
                     complex3264 [:,:,:,:,:,:] jhj,
                     complex3264 [:,:,:,:,:,:] upd):
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
def cycompute_weights(complex3264 [:,:,:,:,:,:,:] r,
                     complex3264 [:,:] cov,
                     complex3264[:,:,:,:,:,:] w,
                     float v):
    """
    This function updates the weights, using the 
    expression w[i] = (v+8)/(v + 2*r[i].T.cov.r[i])
    r: the reisudals
    cov : the weigted covariance matrix
    """

    cdef int d, t, f, aa, ab = 0
    cdef int n_mod, n_tim, n_fre, n_ant
    cdef complex3264 r00, r01, r10, r11, denom
    cdef complex3264 c00, c01, c02, c03, c10, c11, c12, c13, c20, c21, c22, c23, c30, c31, c32, c33

    c00, c01, c02, c03 = cov[0,0], cov[0,1], cov[0,2], cov[0,3]
    c10, c11, c12, c13 = cov[1,0], cov[1,1], cov[1,2], cov[1,3]
    c20, c21, c22, c23 = cov[2,0], cov[2,1], cov[2,2], cov[2,3]
    c30, c31, c32, c33 = cov[3,0], cov[3,1], cov[3,2], cov[3,3]


    n_mod = r.shape[0]
    n_tim = r.shape[1]
    n_fre = r.shape[2]
    n_ant = r.shape[3]

    for i in xrange(n_mod):
        for t in xrange(n_tim):
            for f in xrange(n_fre):
                for aa in xrange(n_ant):
                    for ab in xrange(n_ant):
                        r00, r01, r10, r11 = r[i,t,f,aa,ab,0,0], r[i,t,f,aa,ab,0,1], r[i,t,f,aa,ab,1,0], r[i,t,f,aa,ab,1,1]
                        
                        denom = r00*(c00*r00.conjugate() + c10*r10.conjugate() + c20*r01.conjugate() + c30*r11.conjugate()) + r01*(c02*r00.conjugate() + c12*r10.conjugate() + c22*r01.conjugate() + c32*r11.conjugate()) + r10*(c01*r00.conjugate() + c11*r10.conjugate() + c21*r01.conjugate() + c31*r11.conjugate()) + r11*(c03*r00.conjugate() + c13*r10.conjugate() + c23*r01.conjugate() + c33*r11.conjugate())

                        w[i,t,f,aa,ab,0] = (v+8)/(v + 2*denom)


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
def cyapply_gains(complex3264 [:,:,:,:,:,:,:,:] m,
                  complex3264 [:,:,:,:,:,:] g,
                  complex3264 [:,:,:,:,:,:] gh,
                  int t_int,
                  int f_int):

    """
    Apply the gains to the model array - this is useful in general, but is required for using an 
    arbitrary chain of Jones matrices. NOTE: This will perform the computation in place - be wary of 
    overwriting the original model data.  
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

    for d in xrange(n_dir):
        gd = d%g_dir
        for i in xrange(n_mod):
            for t in xrange(n_tim):
                rr = t/t_int
                for f in xrange(n_fre):
                    rc = f/f_int
                    for aa in xrange(n_ant):
                        for ab in xrange(n_ant):

                            gmtmp1 = g[gd,rr,rc,aa,0,0]*m[d,i,t,f,aa,ab,0,0] + \
                                     g[gd,rr,rc,aa,0,1]*m[d,i,t,f,aa,ab,1,0]

                            gmtmp2 = g[gd,rr,rc,aa,0,0]*m[d,i,t,f,aa,ab,0,1] + \
                                     g[gd,rr,rc,aa,0,1]*m[d,i,t,f,aa,ab,1,1]

                            gmtmp3 = g[gd,rr,rc,aa,1,0]*m[d,i,t,f,aa,ab,0,0] + \
                                     g[gd,rr,rc,aa,1,1]*m[d,i,t,f,aa,ab,1,0]

                            gmtmp4 = g[gd,rr,rc,aa,1,0]*m[d,i,t,f,aa,ab,0,1] + \
                                     g[gd,rr,rc,aa,1,1]*m[d,i,t,f,aa,ab,1,1]

                            m[d,i,t,f,aa,ab,0,0] = \
                                gmtmp1*gh[gd,rr,rc,ab,0,0] + \
                                gmtmp2*gh[gd,rr,rc,ab,1,0]

                            m[d,i,t,f,aa,ab,0,1] = \
                                gmtmp1*gh[gd,rr,rc,ab,0,1] + \
                                gmtmp2*gh[gd,rr,rc,ab,1,1]

                            m[d,i,t,f,aa,ab,1,0] = \
                                gmtmp3*gh[gd,rr,rc,ab,0,0] + \
                                gmtmp4*gh[gd,rr,rc,ab,1,0]

                            m[d,i,t,f,aa,ab,1,1] = \
                                gmtmp3*gh[gd,rr,rc,ab,0,1] + \
                                gmtmp4*gh[gd,rr,rc,ab,1,1]