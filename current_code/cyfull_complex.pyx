from cython.parallel import prange, parallel
import numpy as np
cimport numpy as np
import cython


def compute_jhr(double complex [:,:,:,:,:,:] in1,
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


def compute_Abyb(double complex [:,:,:,:,:,:] in1,
                 double complex [:,:,:,:,:] in2,
                 double complex [:,:,:,:,:,:] out1,
                 int t_int,
                 int f_int):
    """
    NOTE: THIS RIGHT-MULTIPLIES THE ENTRIES OF IN1 BY THE ENTRIES OF IN2.
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


def compute_AbyA(double complex [:,:,:,:,:,:] in1,
                  double complex [:,:,:,:,:,:] in2,
                  double complex [:,:,:,:,:,:] out1):
    """
    NOTE: THIS RIGHT-MULTIPLIES THE ENTRIES OF IN1 BY THE ENTRIES OF IN2.
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

def invert_jhj(double complex [:,:,:,:,:] jhj):
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


def reduce_6d(double complex [:,:,:,:,:,:] in1,
              double complex [:,:,:,:,:,:] out1,
              int t_int,
              int f_int):

    """
    NOTE: THIS RIGHT-MULTIPLIES THE COLUMNS OF IN1 BY THE ENTRIES OF IN2.

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
                    out1[rr,rc,k,l,0,0] = out1[rr,rc,k,l,0,0] + in1[i,j,k,l,0,0]
                    out1[rr,rc,k,l,0,1] = out1[rr,rc,k,l,0,1] + in1[i,j,k,l,0,1]
                    out1[rr,rc,k,l,1,0] = out1[rr,rc,k,l,1,0] + in1[i,j,k,l,1,0]
                    out1[rr,rc,k,l,1,1] = out1[rr,rc,k,l,1,1] + in1[i,j,k,l,1,1]


def compute_update(double complex [:,:,:,:,:] in1,
                   double complex [:,:,:,:,:] in2,
                   double complex [:,:,:,:,:] out1):
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

def compute_bbyA(double complex [:,:,:,:,:] in1,
                 double complex [:,:,:,:,:,:] in2,
                 double complex [:,:,:,:,:,:] out1,
                 int t_int,
                 int f_int):
    """
    NOTE: THIS RIGHT-MULTIPLIES THE ENTRIES OF IN1 BY THE ENTRIES OF IN2.
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

                    out1[i,j,k,l,0,0] = in1[rr,rc,k,0,0] * in2[i,j,k,l,0,0] + \
                                        in1[rr,rc,k,0,1] * in2[i,j,k,l,1,0]

                    out1[i,j,k,l,0,1] = in1[rr,rc,k,0,0] * in2[i,j,k,l,0,1] + \
                                        in1[rr,rc,k,0,1] * in2[i,j,k,l,1,1]

                    out1[i,j,k,l,1,0] = in1[rr,rc,k,1,0] * in2[i,j,k,l,0,0] + \
                                        in1[rr,rc,k,1,1] * in2[i,j,k,l,1,0]

                    out1[i,j,k,l,1,1] = in1[rr,rc,k,1,0] * in2[i,j,k,l,0,1] + \
                                        in1[rr,rc,k,1,1] * in2[i,j,k,l,1,1]