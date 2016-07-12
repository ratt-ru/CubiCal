from cython.parallel import prange, parallel
import numpy as np
cimport numpy as np
import cython


def compute_rgmh(double complex [:,:,:,:,:,:] in1,
                 double complex [:,:,:,:,:] in2,
                 double complex [:,:,:,:,:,:] in3,
                 double complex [:,:] tmp1,
                 double complex [:,:,:,:,:] out1,
                 int t_int,
                 int f_int):

    """
    NOTE: THIS RIGHT-MULTIPLIES THE COLUMNS OF IN1 BY THE ENTRIES OF IN2.

    :param in1:
    :param in2:
    :param out1:
    :return:
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

                    tmp1[0,0] = in1[i,j,k,l,0,0] * in2[rr,rc,l,0,0] +\
                                in1[i,j,k,l,0,1] * in2[rr,rc,l,1,0]

                    tmp1[0,1] = in1[i,j,k,l,0,0] * in2[rr,rc,l,0,1] +\
                                in1[i,j,k,l,0,1] * in2[rr,rc,l,1,1]

                    tmp1[1,0] = in1[i,j,k,l,1,0] * in2[rr,rc,l,0,0] +\
                                in1[i,j,k,l,1,1] * in2[rr,rc,l,1,0]

                    tmp1[1,1] = in1[i,j,k,l,1,0] * in2[rr,rc,l,0,1] +\
                                in1[i,j,k,l,1,1] * in2[rr,rc,l,1,1]

                    out1[i,j,k,0,0] = out1[i,j,k,0,0] + \
                                     (tmp1[0,0] * in3[i,j,k,l,0,0]) + \
                                     (tmp1[0,1] * in3[i,j,k,l,0,1])

                    out1[i,j,k,0,1] = out1[i,j,k,0,1] + \
                                     (tmp1[0,0] * in3[i,j,k,l,1,0]) + \
                                     (tmp1[0,1] * in3[i,j,k,l,1,1])

                    out1[i,j,k,1,0] = out1[i,j,k,1,0] + \
                                     (tmp1[1,0] * in3[i,j,k,l,0,0]) + \
                                     (tmp1[1,1] * in3[i,j,k,l,0,1])

                    out1[i,j,k,1,1] = out1[i,j,k,1,1] + \
                                     (tmp1[1,0] * in3[i,j,k,l,1,0]) + \
                                     (tmp1[1,1] * in3[i,j,k,l,1,1])


def compute_jhj(double complex [:,:,:,:,:,:] in1,
                double complex [:,:,:,:,:] in2,
                double complex [:,:,:,:,:] out1,
                int t_int,
                int f_int):
    """
    NOTE: THIS RIGHT-MULTIPLIES THE ENTRIES OF IN1 BY THE ENTRIES OF IN2.
    """

    cdef int i,j,k,l,rr,rc = 0
    cdef double complex tmp1

    cdef long new_shape[6]

    new_shape[:] = in1.shape

    for i in xrange(new_shape[0]):
        rr = i//t_int
        for j in xrange(new_shape[1]):
            rc = j//f_int
            for k in xrange(new_shape[2]):
                for l in xrange(k+1, new_shape[3]):

                    out1[i,j,k,0,0] = out1[i,j,k,0,0] + \
                                      in1[i,j,k,l,0,0] * in2[i,j,k,0,0] + \
                                      in1[i,j,k,l,0,1] * in2[i,j,k,0,1]

                    out1[i,j,k,0,1] = out1[i,j,k,0,1] + \
                                      in1[i,j,k,l,0,0] * in2[i,j,k,1,0] + \
                                      in1[i,j,k,l,0,1] * in2[i,j,k,1,1]

                    out1[i,j,k,1,0] = out1[i,j,k,1,0] + \
                                      in1[i,j,k,l,1,0] * in2[i,j,k,0,0] + \
                                      in1[i,j,k,l,1,1] * in2[i,j,k,0,1]

                    out1[i,j,k,1,1] = out1[i,j,k,1,1] + \
                                      in1[i,j,k,l,1,0] * in2[i,j,k,1,0] + \
                                      in1[i,j,k,l,1,1] * in2[i,j,k,1,1]