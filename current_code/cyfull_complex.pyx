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

@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
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

@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
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


@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
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

@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
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

@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
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

@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
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

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
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

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def model_reduce(complex3264 [:,:,:,:,:,:] in1,
                 complex3264 [:,:,:,:,:,:] out1,
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
                    out1[rr,rc,k,l,0,0] = out1[rr,rc,k,l,0,0] + in1[i,j,k,l,0,0]
                    out1[rr,rc,k,l,0,1] = out1[rr,rc,k,l,0,1] + in1[i,j,k,l,0,1]
                    out1[rr,rc,k,l,1,0] = out1[rr,rc,k,l,1,0] + in1[i,j,k,l,1,0]
                    out1[rr,rc,k,l,1,1] = out1[rr,rc,k,l,1,1] + in1[i,j,k,l,1,1]