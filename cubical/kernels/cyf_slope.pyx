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

@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def cycompute_tmp_jhj(complex3264 [:,:,:,:,:,:,:,:] m,
                      complex3264 [:,:,:,:,:,:,:] jhj):

    """
    This reduces the dimension of in1 to match out1. This is achieved by a
    summation of blocks of dimension (t_int, f_int).
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
                            
                            jhj[d,i,t,f,aa,0,0] = jhj[d,i,t,f,aa,0,0] + \
                                                  m[d,i,t,f,aa,ab,0,0]*m[d,i,t,f,ab,aa,0,0] + \
                                                  m[d,i,t,f,aa,ab,0,1]*m[d,i,t,f,ab,aa,1,0]
                            jhj[d,i,t,f,aa,1,1] = jhj[d,i,t,f,aa,1,1] + \
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
    This reduces the dimension of in1 to match out1. This is achieved by a
    summation of blocks of dimension (t_int, f_int).
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
                     np.uint16_t [:,:,:,:] flags,
                     float eps, 
                     int flagbit):

    cdef int d, t, f, c, aa = 0
    cdef int n_dir, n_tim, n_fre, n_ant
    cdef int flag_count = 0
    cdef float3264 det00, det11 = 0

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

                            flags[d,t,f,aa] = flagbit
                            flag_count += 1

                        else:

                            det = 1/det

                            jhjinv[d,t,f,aa,0,c,c] = det*jhj[d,t,f,aa,2,c,c]

                            jhjinv[d,t,f,aa,1,c,c] = -1*det*jhj[d,t,f,aa,1,c,c]

                            jhjinv[d,t,f,aa,2,c,c] = det*jhj[d,t,f,aa,0,c,c]

    return flag_count

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
    This reduces the dimension of in1 to match out1. This is achieved by a
    summation of blocks of dimension (t_int, f_int).
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
    This reduces the dimension of in1 to match out1. This is achieved by a
    summation of blocks of dimension (t_int, f_int).
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
    This reduces the dimension of in1 to match out1. This is achieved by a
    summation of blocks of dimension (t_int, f_int).
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
    NOTE: THIS RIGHT-MULTIPLIES THE ENTRIES OF IN1 BY THE ENTRIES OF IN2.
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
    This reduces the dimension of in1 to match out1. This is achieved by a
    summation of blocks of dimension (t_int, f_int).
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
    This reduces the dimension of in1 to match out1. This is achieved by a
    summation of blocks of dimension (t_int, f_int).
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
    This reduces the dimension of in1 to match out1. This is achieved by a
    summation of blocks of dimension (t_int, f_int).
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
                        g[d,t,f,aa,c,c] = np.exp(1j*(fs[f]*param[d,rr,rc,aa,0,c,c] + 
                                                           param[d,rr,rc,aa,1,c,c] ))
