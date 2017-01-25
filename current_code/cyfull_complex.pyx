from cython.parallel import prange, parallel
import numpy as np
cimport numpy as np
import cython

ctypedef fused complex3264:
    np.complex64_t
    np.complex128_t

# @cython.cdivision(True)
# @cython.wraparound(False)
# @cython.boundscheck(False)
# @cython.nonecheck(False)
def compute_jhr(complex3264 [:,:,:,:,:,:] in1,
                complex3264 [:,:,:,:,:] in2,
                complex3264 [:,:,:,:,:,:] in3,
                complex3264 [:,:] tmp1,
                complex3264 [:,:,:,:,:] out1,
                int t_int,
                int f_int):

    """
    NOTE: THIS RIGHT-MULTIPLIES THE COLUMNS OF IN1 BY THE ENTRIES OF IN2.

    :param in1:
    :param in2:
    :param out1:
    :return:
    """

    cdef int i, j, k, l, rr, rc = 0

    cdef long new_shape[6]

    new_shape[:] = in1.shape

    for i in xrange(new_shape[0]):
        rr = i//t_int
        for j in xrange(new_shape[1]):
            rc = j//f_int
            for k in xrange(new_shape[2]):
                for l in xrange(new_shape[3]):

                    tmp1[0,0] = in1[i,j,k,l,0,0] * in2[rr,rc,l,0,0] +\
                                in1[i,j,k,l,0,1] * in2[rr,rc,l,1,0]

                    tmp1[0,1] = in1[i,j,k,l,0,0] * in2[rr,rc,l,0,1] +\
                                in1[i,j,k,l,0,1] * in2[rr,rc,l,1,1]

                    tmp1[1,0] = in1[i,j,k,l,1,0] * in2[rr,rc,l,0,0] +\
                                in1[i,j,k,l,1,1] * in2[rr,rc,l,1,0]

                    tmp1[1,1] = in1[i,j,k,l,1,0] * in2[rr,rc,l,0,1] +\
                                in1[i,j,k,l,1,1] * in2[rr,rc,l,1,1]

                    out1[i,j,k,0,0] = out1[i,j,k,0,0] + \
                                     (tmp1[0,0] * in3[i,j,l,k,0,0]) + \
                                     (tmp1[0,1] * in3[i,j,l,k,1,0])

                    out1[i,j,k,0,1] = out1[i,j,k,0,1] + \
                                     (tmp1[0,0] * in3[i,j,l,k,0,1]) + \
                                     (tmp1[0,1] * in3[i,j,l,k,1,1])

                    out1[i,j,k,1,0] = out1[i,j,k,1,0] + \
                                     (tmp1[1,0] * in3[i,j,l,k,0,0]) + \
                                     (tmp1[1,1] * in3[i,j,l,k,1,0])

                    out1[i,j,k,1,1] = out1[i,j,k,1,1] + \
                                     (tmp1[1,0] * in3[i,j,l,k,0,1]) + \
                                     (tmp1[1,1] * in3[i,j,l,k,1,1])

# @cython.cdivision(True)
# @cython.wraparound(False)
# @cython.boundscheck(False)
# @cython.nonecheck(False)
def compute_Abyb(complex3264 [:,:,:,:,:,:] in1,
                 complex3264 [:,:,:,:,:] in2,
                 complex3264 [:,:,:,:,:,:] out1,
                 int t_int,
                 int f_int):
    """
    This takes the dot product of the elements of matrix in1 with the elements
    of a vector in2. Note that the elements of both are 2-by-2 blocks.
    Broadcasting is done along rows.
    """

    cdef int i,j,k,l,rr,rc = 0

    cdef long new_shape[6]

    new_shape[:] = in1.shape

    for i in xrange(new_shape[0]):
        rr = i//t_int
        for j in xrange(new_shape[1]):
            rc = j//f_int
            for k in xrange(new_shape[2]):
                for l in xrange(new_shape[3]):

                    out1[i,j,k,l,0,0] = in1[i,j,k,l,0,0] * in2[rr,rc,l,0,0] + \
                                        in1[i,j,k,l,0,1] * in2[rr,rc,l,1,0]

                    out1[i,j,k,l,0,1] = in1[i,j,k,l,0,0] * in2[rr,rc,l,0,1] + \
                                        in1[i,j,k,l,0,1] * in2[rr,rc,l,1,1]

                    out1[i,j,k,l,1,0] = in1[i,j,k,l,1,0] * in2[rr,rc,l,0,0] + \
                                        in1[i,j,k,l,1,1] * in2[rr,rc,l,1,0]

                    out1[i,j,k,l,1,1] = in1[i,j,k,l,1,0] * in2[rr,rc,l,0,1] + \
                                        in1[i,j,k,l,1,1] * in2[rr,rc,l,1,1]

# @cython.cdivision(True)
# @cython.wraparound(False)
# @cython.boundscheck(False)
# @cython.nonecheck(False)
def compute_AbyA(complex3264 [:,:,:,:,:,:] in1,
                 complex3264 [:,:,:,:,:,:] in2,
                 complex3264 [:,:,:,:,:,:] out1):
    """
    This takes the dot product of the elements of matrix in1 with the elements
    of matrix in2. Note that the elements of both are 2-by-2 blocks.
    """

    cdef int i,j,k,l = 0

    cdef long new_shape[6]

    new_shape[:] = in1.shape

    for i in xrange(new_shape[0]):
        for j in xrange(new_shape[1]):
            for k in xrange(new_shape[2]):
                for l in xrange(new_shape[3]):

                    out1[i,j,k,l,0,0] = in1[i,j,k,l,0,0] * in2[i,j,l,k,0,0] + \
                                        in1[i,j,k,l,0,1] * in2[i,j,l,k,1,0]

                    out1[i,j,k,l,0,1] = in1[i,j,k,l,0,0] * in2[i,j,l,k,0,1] + \
                                        in1[i,j,k,l,0,1] * in2[i,j,l,k,1,1]

                    out1[i,j,k,l,1,0] = in1[i,j,k,l,1,0] * in2[i,j,l,k,0,0] + \
                                        in1[i,j,k,l,1,1] * in2[i,j,l,k,1,0]

                    out1[i,j,k,l,1,1] = in1[i,j,k,l,1,0] * in2[i,j,l,k,0,1] + \
                                        in1[i,j,k,l,1,1] * in2[i,j,l,k,1,1]


# @cython.cdivision(True)
# @cython.wraparound(False)
# @cython.boundscheck(False)
# @cython.nonecheck(False)
def invert_jhj(complex3264 [:,:,:,:,:] jhj):
    """
    NOTE: THIS RIGHT-MULTIPLIES THE ENTRIES OF IN1 BY THE ENTRIES OF IN2.
    """

    cdef int i,j,k = 0
    cdef double complex denom, store = 0

    cdef long new_shape[5]

    new_shape[:] = jhj.shape

    for i in xrange(new_shape[0]):
        for j in xrange(new_shape[1]):
            for k in xrange(new_shape[2]):

                denom = jhj[i,j,k,0,0] * jhj[i,j,k,1,1] - \
                        jhj[i,j,k,0,1] * jhj[i,j,k,1,0]

                if denom==0:
                    denom = 1

                store = jhj[i,j,k,0,0]

                jhj[i,j,k,0,0] = jhj[i,j,k,1,1]/denom
                jhj[i,j,k,1,1] = store/denom
                jhj[i,j,k,0,1] = -1 * jhj[i,j,k,0,1]/denom
                jhj[i,j,k,1,0] = -1 * jhj[i,j,k,1,0]/denom

# @cython.cdivision(True)
# @cython.wraparound(False)
# @cython.boundscheck(False)
# @cython.nonecheck(False)
def reduce_6d(complex3264 [:,:,:,:,:,:] in1,
              complex3264 [:,:,:,:,:,:] out1,
              int t_int,
              int f_int):

    """
    Summation over the time and frequency intervals for a 6D matrix. This
    does not reduce its dimension - merely squashes it into fewer elements.
    """

    cdef int i, j, k, l, rr, rc = 0

    cdef long new_shape[6]

    new_shape[:] = in1.shape

    for i in xrange(new_shape[0]):
        rr = i//t_int
        for j in xrange(new_shape[1]):
            rc = j//f_int
            for k in xrange(new_shape[2]):
                for l in xrange(new_shape[3]):
                    out1[rr,rc,k,l,0,0] = out1[rr,rc,k,l,0,0] + in1[i,j,k,l,0,0]
                    out1[rr,rc,k,l,0,1] = out1[rr,rc,k,l,0,1] + in1[i,j,k,l,0,1]
                    out1[rr,rc,k,l,1,0] = out1[rr,rc,k,l,1,0] + in1[i,j,k,l,1,0]
                    out1[rr,rc,k,l,1,1] = out1[rr,rc,k,l,1,1] + in1[i,j,k,l,1,1]

# @cython.cdivision(True)
# @cython.wraparound(False)
# @cython.boundscheck(False)
# @cython.nonecheck(False)
def compute_update(complex3264 [:,:,:,:,:] in1,
                   complex3264 [:,:,:,:,:] in2,
                   complex3264 [:,:,:,:,:] out1):
    """
    NOTE: THIS RIGHT-MULTIPLIES THE ENTRIES OF IN1 BY THE ENTRIES OF IN2.
    """

    cdef int i,j,k = 0

    cdef long new_shape[5]

    new_shape[:] = out1.shape

    for i in xrange(new_shape[0]):
        for j in xrange(new_shape[1]):
            for k in xrange(new_shape[2]):

                out1[i,j,k,0,0] = in1[i,j,k,0,0] * in2[i,j,k,0,0] + \
                                  in1[i,j,k,0,1] * in2[i,j,k,1,0]

                out1[i,j,k,0,1] = in1[i,j,k,0,0] * in2[i,j,k,0,1] + \
                                  in1[i,j,k,0,1] * in2[i,j,k,1,1]

                out1[i,j,k,1,0] = in1[i,j,k,1,0] * in2[i,j,k,0,0] + \
                                  in1[i,j,k,1,1] * in2[i,j,k,1,0]

                out1[i,j,k,1,1] = in1[i,j,k,1,0] * in2[i,j,k,0,1] + \
                                  in1[i,j,k,1,1] * in2[i,j,k,1,1]

# @cython.cdivision(True)
# @cython.wraparound(False)
# @cython.boundscheck(False)
# @cython.nonecheck(False)
def compute_bbyA(complex3264 [:,:,:,:,:] in1,
                 complex3264 [:,:,:,:,:,:] in2,
                 complex3264 [:,:,:,:,:,:] out1,
                 int t_int,
                 int f_int):
    """
    NOTE: THIS RIGHT-MULTIPLIES THE ENTRIES OF IN1 BY THE ENTRIES OF IN2.
    """

    cdef int i,j,k,l,rr,rc = 0

    cdef long new_shape[6]

    new_shape[:] = in2.shape

    for i in xrange(new_shape[0]):
        rr = i//t_int
        for j in xrange(new_shape[1]):
            rc = j//f_int
            for k in xrange(new_shape[2]):
                for l in xrange(new_shape[3]):

                    out1[i,j,k,l,0,0] = in1[rr,rc,k,0,0] * in2[i,j,k,l,0,0] + \
                                        in1[rr,rc,k,0,1] * in2[i,j,k,l,1,0]

                    out1[i,j,k,l,0,1] = in1[rr,rc,k,0,0] * in2[i,j,k,l,0,1] + \
                                        in1[rr,rc,k,0,1] * in2[i,j,k,l,1,1]

                    out1[i,j,k,l,1,0] = in1[rr,rc,k,1,0] * in2[i,j,k,l,0,0] + \
                                        in1[rr,rc,k,1,1] * in2[i,j,k,l,1,0]

                    out1[i,j,k,l,1,1] = in1[rr,rc,k,1,0] * in2[i,j,k,l,0,1] + \
                                        in1[rr,rc,k,1,1] * in2[i,j,k,l,1,1]


# @cython.wraparound(False)
# @cython.boundscheck(False)
# @cython.nonecheck(False)
def interval_reduce(complex3264 [:,:,:,:,:] in1,
                    complex3264 [:,:,:,:,:] out1,
                    int t_int,
                    int f_int):

    """
    NOTE: THIS RIGHT-MULTIPLIES THE COLUMNS OF IN1 BY THE ENTRIES OF IN2.

    :param in1:
    :param in2:
    :param out1:
    :return:
    """

    cdef int i, j, k, l, rr, rc = 0

    cdef long new_shape[6]

    new_shape[:] = in1.shape

    for i in xrange(new_shape[0]):
        rr = i//t_int
        for j in xrange(new_shape[1]):
            rc = j//f_int
            for k in xrange(new_shape[2]):
                out1[rr,rc,k,0,0] = out1[rr,rc,k,0,0] + in1[i,j,k,0,0]
                out1[rr,rc,k,0,1] = out1[rr,rc,k,0,1] + in1[i,j,k,0,1]
                out1[rr,rc,k,1,0] = out1[rr,rc,k,1,0] + in1[i,j,k,1,0]
                out1[rr,rc,k,1,1] = out1[rr,rc,k,1,1] + in1[i,j,k,1,1]

# @cython.cdivision(True)
# @cython.wraparound(False)
# @cython.boundscheck(False)
# @cython.nonecheck(False)
def model_reduce(complex3264 [:,:,:,:,:,:,:] in1,
                 complex3264 [:,:,:,:,:,:,:] out1,
                 int t_int,
                 int f_int):

    """
    This reduces the dimension of in1 to match out1. This is achieved by a
    summation of blocks of dimension (t_int, f_int).
    """

    cdef int d, t, f, aa, ab, rr, rc
    cdef int n_dir, n_tim, n_fre, n_ant
    
    n_dir = in1.shape[0]
    n_tim = in1.shape[1]
    n_fre = in1.shape[2]
    n_ant = in1.shape[3]

    for d in xrange(n_dir):
        for t in xrange(n_tim):
            rr = t/t_int
            for f in xrange(n_fre):
                rc = f/f_int
                for aa in xrange(n_ant):
                    for ab in xrange(n_ant):
                        out1[d,rr,rc,aa,ab,0,0] = out1[d,rr,rc,aa,ab,0,0] + \
                                                   in1[d,t,f,aa,ab,0,0]
                        out1[d,rr,rc,aa,ab,0,1] = out1[d,rr,rc,aa,ab,0,1] + \
                                                   in1[d,t,f,aa,ab,0,1]
                        out1[d,rr,rc,aa,ab,1,0] = out1[d,rr,rc,aa,ab,1,0] + \
                                                   in1[d,t,f,aa,ab,1,0]
                        out1[d,rr,rc,aa,ab,1,1] = out1[d,rr,rc,aa,ab,1,1] + \
                                                   in1[d,t,f,aa,ab,1,1]

# @cython.cdivision(True)
# @cython.wraparound(False)
# @cython.boundscheck(False)
# @cython.nonecheck(False)
def cycompute_residual(complex3264 [:,:,:,:,:,:,:] m,
                       complex3264 [:,:,:,:,:,:] g,
                       complex3264 [:,:,:,:,:,:] gh,
                       complex3264 [:,:,:,:,:,:] r):

    """
    This reduces the dimension of in1 to match out1. This is achieved by a
    summation of blocks of dimension (t_int, f_int).
    """

    cdef int d, t, f, aa, ab
    cdef int n_dir, n_tim, n_fre, n_ant

    n_dir = m.shape[0]
    n_tim = m.shape[1]
    n_fre = m.shape[2]
    n_ant = m.shape[3]

    for d in xrange(n_dir):
        for t in xrange(n_tim):
            for f in xrange(n_fre):
                for aa in xrange(n_ant):
                    for ab in xrange(n_ant):
                        r[t,f,aa,ab,0,0] = r[t,f,aa,ab,0,0] + \
                        g[d,t,f,aa,0,0]*m[d,t,f,aa,ab,0,0]*gh[d,t,f,ab,0,0] + \
                        g[d,t,f,aa,0,1]*m[d,t,f,aa,ab,1,0]*gh[d,t,f,ab,0,0] + \
                        g[d,t,f,aa,0,0]*m[d,t,f,aa,ab,0,1]*gh[d,t,f,ab,1,0] + \
                        g[d,t,f,aa,0,1]*m[d,t,f,aa,ab,1,1]*gh[d,t,f,ab,1,0]

                        r[t,f,aa,ab,0,1] = r[t,f,aa,ab,0,1] + \
                        g[d,t,f,aa,0,0]*m[d,t,f,aa,ab,0,0]*gh[d,t,f,ab,0,1] + \
                        g[d,t,f,aa,0,1]*m[d,t,f,aa,ab,1,0]*gh[d,t,f,ab,0,1] + \
                        g[d,t,f,aa,0,0]*m[d,t,f,aa,ab,0,1]*gh[d,t,f,ab,1,1] + \
                        g[d,t,f,aa,0,1]*m[d,t,f,aa,ab,1,1]*gh[d,t,f,ab,1,1]

                        r[t,f,aa,ab,1,0] = r[t,f,aa,ab,1,0] + \
                        g[d,t,f,aa,1,0]*m[d,t,f,aa,ab,0,0]*gh[d,t,f,ab,0,0] + \
                        g[d,t,f,aa,1,1]*m[d,t,f,aa,ab,1,0]*gh[d,t,f,ab,0,0] + \
                        g[d,t,f,aa,1,0]*m[d,t,f,aa,ab,0,1]*gh[d,t,f,ab,1,0] + \
                        g[d,t,f,aa,1,1]*m[d,t,f,aa,ab,1,1]*gh[d,t,f,ab,1,0]

                        r[t,f,aa,ab,1,1] = r[t,f,aa,ab,1,1] + \
                        g[d,t,f,aa,1,0]*m[d,t,f,aa,ab,0,0]*gh[d,t,f,ab,0,1] + \
                        g[d,t,f,aa,1,1]*m[d,t,f,aa,ab,1,0]*gh[d,t,f,ab,0,1] + \
                        g[d,t,f,aa,1,0]*m[d,t,f,aa,ab,0,1]*gh[d,t,f,ab,1,1] + \
                        g[d,t,f,aa,1,1]*m[d,t,f,aa,ab,1,1]*gh[d,t,f,ab,1,1]