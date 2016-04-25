from cython.parallel import prange, parallel
import numpy as np
cimport numpy as np
import cython

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def compute_rgmh(double complex [:,:,:,:,:,:] in1,
                 double complex [:,:,:,:,:] in2,
                 double complex [:,:,:,:,:,:] in3,
                 double complex [:,:] tmp1,
                 double complex [:,:,:,:,:] out1):

    """
    NOTE: THIS RIGHT-MULTIPLIES THE COLUMNS OF IN1 BY THE ENTRIES OF IN2.

    :param in1:
    :param in2:
    :param out1:
    :return:
    """

    cdef int i,j,k,l = 0

    cdef long new_shape[6]

    new_shape[:] = in1.shape

    for i in xrange(new_shape[0]):
        for j in xrange(new_shape[1]):
            for k in xrange(new_shape[2]):
                for l in xrange(new_shape[3]):

                    tmp1[0,0] = in1[i,j,k,l,0,0] * in2[i,j,l,0,0]

                    tmp1[0,1] = in1[i,j,k,l,0,1] * in2[i,j,l,1,1]

                    tmp1[1,0] = in1[i,j,k,l,1,0] * in2[i,j,l,0,0]

                    tmp1[1,1] = in1[i,j,k,l,1,1] * in2[i,j,l,1,1]

                    out1[i,j,k,0,0] = out1[i,j,k,0,0] + (tmp1[0,0]*
                                         in3[i,j,k,l,0,0]) + \
                                        (tmp1[0,1]*
                                         in3[i,j,k,l,0,1])

                    out1[i,j,k,0,1] = out1[i,j,k,0,1] + (tmp1[0,0]*
                                         in3[i,j,k,l,1,0]) + \
                                        (tmp1[0,1]*
                                         in3[i,j,k,l,1,1])

                    out1[i,j,k,1,0] = out1[i,j,k,1,0] + (tmp1[1,0]*
                                         in3[i,j,k,l,0,0]) + \
                                        (tmp1[1,1]*
                                         in3[i,j,k,l,0,1])

                    out1[i,j,k,1,1] = out1[i,j,k,1,1] + (tmp1[1,0]*
                                         in3[i,j,k,l,1,0]) + \
                                        (tmp1[1,1]*
                                         in3[i,j,k,l,1,1])

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def compute_ghirmgh(double complex [:,:,:,:,:] in1,
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

                out1[i,j,k,0,0] = in1[i,j,k,0,0] * in2[i,j,k,0,0]
                out1[i,j,k,1,0] = in1[i,j,k,1,1] * in2[i,j,k,1,1]

@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def compute_jhj(double complex [:,:,:,:,:,:] in1,
                double [:,:,:,:,:] out1):
    """
    NOTE: THIS RIGHT-MULTIPLIES THE ENTRIES OF IN1 BY THE ENTRIES OF IN2.
    """

    cdef int i,j,k,l = 0
    cdef double tmp1

    cdef long new_shape[6]

    new_shape[:] = in1.shape

    for i in xrange(new_shape[0]):
        for j in xrange(new_shape[1]):
            for k in xrange(new_shape[2]):
                for l in xrange(new_shape[3]):

                    tmp1 = ((in1[i,j,k,l,1,0] * in1[i,j,l,k,1,0])
                          + (in1[i,j,k,l,0,1] * in1[i,j,l,k,0,1])).real

                    out1[i,j,k,0,0] = out1[i,j,k,0,0] + tmp1 + 2 * \
                                     (in1[i,j,k,l,0,0] * in1[i,j,l,k,0,0]).real

                    out1[i,j,k,1,1] = out1[i,j,k,1,1] + tmp1 + 2 * \
                                     (in1[i,j,k,l,1,1] * in1[i,j,l,k,1,1]).real

                if out1[i,j,k,0,0] != 0:
                    out1[i,j,k,0,0] = 1/out1[i,j,k,0,0]

                if out1[i,j,k,1,1] != 0:
                    out1[i,j,k,1,1] = 1/out1[i,j,k,1,1]

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def compute_update(double [:,:,:,:,:] in1,
                   double [:,:,:,:,:] in2,
                   double [:,:,:,:,:] out1):
    """
    NOTE: THIS RIGHT-MULTIPLIES THE ENTRIES OF IN1 BY THE ENTRIES OF IN2.
    """

    cdef int i,j,k = 0

    cdef long new_shape[5]

    new_shape[:] = out1.shape

    for i in xrange(new_shape[0]):
        for j in xrange(new_shape[1]):
            for k in xrange(new_shape[2]):

                out1[i,j,k,0,0] = in1[i,j,k,0,0] * in2[i,j,k,0,0]
                out1[i,j,k,1,0] = in1[i,j,k,1,1] * in2[i,j,k,1,0]


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def apply_gains(double complex [:,:,:,:,:] in1,
                double complex [:,:,:,:,:] in2,
                double complex [:,:,:,:,:,:] in3,
                double complex [:,:,:,:,:,:] out1):
    """
    NOTE: THIS RIGHT-MULTIPLIES THE ENTRIES OF IN1 BY THE ENTRIES OF IN2.
    """

    cdef int i,j,k,l = 0

    cdef long new_shape[6]

    new_shape[:] = out1.shape

    for i in xrange(new_shape[0]):
        for j in xrange(new_shape[1]):
            for k in xrange(new_shape[2]):
                for l in xrange(new_shape[3]):
                    out1[i,j,k,l,0,0] = in1[i,j,k,0,0] * in3[i,j,k,l,0,0]
                    out1[i,j,k,l,0,1] = in1[i,j,k,0,0] * in3[i,j,k,l,0,1]
                    out1[i,j,k,l,1,0] = in1[i,j,k,1,1] * in3[i,j,k,l,1,0]
                    out1[i,j,k,l,1,1] = in1[i,j,k,1,1] * in3[i,j,k,l,1,1]

                    out1[i,j,k,l,0,0] = out1[i,j,k,l,0,0] * in2[i,j,l,0,0]
                    out1[i,j,k,l,0,1] = out1[i,j,k,l,0,1] * in2[i,j,l,1,1]
                    out1[i,j,k,l,1,0] = out1[i,j,k,l,1,0] * in2[i,j,l,0,0]
                    out1[i,j,k,l,1,1] = out1[i,j,k,l,1,1] * in2[i,j,l,1,1]

# @cython.wraparound(False)
# @cython.boundscheck(False)
# @cython.nonecheck(False)
# def compute_rg(double complex [:,:,:,:,:,:] in1,
#                double complex [:,:,:,:,:] in2,
#                double complex [:,:,:,:,:,:] out1):
#
#     """
#     NOTE: THIS RIGHT-MULTIPLIES THE COLUMNS OF IN1 BY THE ENTRIES OF IN2.
#
#     :param in1:
#     :param in2:
#     :param out1:
#     :return:
#     """
#
#     cdef int i,j,k,l = 0
#
#     cdef long new_shape[6]
#
#     new_shape[:] = out1.shape
#
#     for i in xrange(new_shape[0]):
#         for j in xrange(new_shape[1]):
#             for k in xrange(new_shape[2]):
#                 for l in xrange(new_shape[3]):
#
#                     out1[i,j,k,l,0,0] = (in1[i,j,k,l,0,0]*
#                                        in2[i,j,l,0,0]) + \
#                                       (in1[i,j,k,l,0,1]*
#                                         in2[i,j,l,1,0])
#
#                     out1[i,j,k,l,0,1] = (in1[i,j,k,l,0,0]*
#                                        in2[i,j,l,0,1]) + \
#                                       (in1[i,j,k,l,0,1]*
#                                         in2[i,j,l,1,1])
#
#                     out1[i,j,k,l,1,0] = (in1[i,j,k,l,1,0]*
#                                        in2[i,j,l,0,0]) + \
#                                       (in1[i,j,k,l,1,1]*
#                                         in2[i,j,l,1,0])
#
#                     out1[i,j,k,l,1,1] = (in1[i,j,k,l,1,0]*
#                                        in2[i,j,l,0,1]) + \
#                                       (in1[i,j,k,l,1,1]*
#                                         in2[i,j,l,1,1])
#
# @cython.wraparound(False)
# @cython.boundscheck(False)
# @cython.nonecheck(False)
# def compute_rgmh(double complex [:,:,:,:,:,:] in1,
#                  double complex [:,:,:,:,:,:] in2,
#                  double complex [:,:,:,:,:,:] out1):
#
#     """
#     NOTE: THIS RIGHT-MULTIPLIES THE ENTRIES OF IN1 BY THE ENTRIES OF IN2.
#     """
#
#     cdef int i,j,k,l = 0
#
#     cdef long new_shape[6]
#
#     new_shape[:] = out1.shape
#
#     for i in xrange(new_shape[0]):
#         for j in xrange(new_shape[1]):
#             for k in xrange(new_shape[2]):
#                 for l in xrange(new_shape[3]):
#
#                     out1[i,j,k,l,0,0] = (in1[i,j,k,l,0,0]*
#                                          in2[i,j,k,l,0,0]) + \
#                                         (in1[i,j,k,l,0,1]*
#                                          in2[i,j,k,l,0,1])
#
#                     out1[i,j,k,l,0,1] = (in1[i,j,k,l,0,0]*
#                                          in2[i,j,k,l,1,0]) + \
#                                         (in1[i,j,k,l,0,1]*
#                                          in2[i,j,k,l,1,1])
#
#                     out1[i,j,k,l,1,0] = (in1[i,j,k,l,1,0]*
#                                          in2[i,j,k,l,0,0]) + \
#                                         (in1[i,j,k,l,1,1]*
#                                          in2[i,j,k,l,0,1])
#
#                     out1[i,j,k,l,1,1] = (in1[i,j,k,l,1,0]*
#                                          in2[i,j,k,l,1,0]) + \
#                                         (in1[i,j,k,l,1,1]*
#                                          in2[i,j,k,l,1,1])
#
#
#
# @cython.wraparound(False)
# @cython.boundscheck(False)
# @cython.nonecheck(False)
# def compute_ghi(double complex [:,:,:,:,:] in1,
#                 double [:,:] in2,
#                 double complex [:,:,:,:,:] out1):
#     """
#     NOTE: THIS RIGHT-MULTIPLIES THE ENTRIES OF IN1 BY THE ENTRIES OF IN2.
#     """
#
#     cdef int i,j,k = 0
#
#     cdef long new_shape[5]
#
#     new_shape[:] = out1.shape
#
#     for i in xrange(new_shape[0]):
#         for j in xrange(new_shape[1]):
#             for k in xrange(new_shape[2]):
#
#                 out1[i,j,k,0,0] = in1[i,j,k,0,0]
#                 out1[i,j,k,1,3] = in1[i,j,k,1,1]
#
# @cython.wraparound(False)
# @cython.boundscheck(False)
# @cython.nonecheck(False)
# def compute_ghirmgh2(double complex [:,:,:,:,:] in1,
#                     double complex [:,:,:,:,:] in2,
#                     double complex [:,:,:,:,:] out1):
#
#     """
#     NOTE: THIS RIGHT-MULTIPLIES THE ROWS OF IN1 BY THE ENTRIES OF IN2.
#
#     :param in1:
#     :param in2:
#     :param out1:
#     :return:
#     """
#
#     cdef int i,j,k = 0
#
#     cdef long new_shape[5]
#
#     new_shape[:] = out1.shape
#
#     for i in xrange(new_shape[0]):
#         for j in xrange(new_shape[1]):
#             for k in xrange(new_shape[2]):
#                 out1[i,j,k,0,0] = in1[i,j,k,0,0]*in2[i,j,k,0,0]
#                 out1[i,j,k,1,0] = in1[i,j,k,1,3]*in2[i,j,k,1,1]

    # with nogil, parallel(num_threads=4):
    #     for i in prange(new_shape[0], schedule="dynamic"):
    #         for j in xrange(new_shape[1]):
    #             for k in xrange(new_shape[2]):
    #                 for l in xrange(new_shape[3]):
    #
    #                     out1[i,j,k,l,0,0] = (in1[i,j,k,l,0,0]*
    #                                        in2[i,j,k,0,0]) + \
    #                                       (in1[i,j,k,l,0,1]*
    #                                         in2[i,j,k,1,0])
    #
    #                     out1[i,j,k,l,0,1] = (in1[i,j,k,l,0,0]*
    #                                        in2[i,j,k,0,1]) + \
    #                                       (in1[i,j,k,l,0,1]*
    #                                         in2[i,j,k,1,1])
    #
    #                     out1[i,j,k,l,1,0] = (in1[i,j,k,l,1,0]*
    #                                        in2[i,j,k,0,0]) + \
    #                                       (in1[i,j,k,l,1,1]*
    #                                         in2[i,j,k,1,0])
    #
    #                     out1[i,j,k,l,1,1] = (in1[i,j,k,l,1,0]*
    #                                        in2[i,j,k,0,1]) + \
    #                                       (in1[i,j,k,l,1,1]*
    #                                         in2[i,j,k,1,1])
