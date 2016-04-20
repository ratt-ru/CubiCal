from cython.parallel import prange, parallel
import numpy as np
cimport numpy as np
import cython

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def compute_jhr(double complex [:,:,:,:,:,:] in1,
                double complex [:,:,:,:,:,:] out1,
                double complex [:,:,:,:,:] in2):

    cdef int i,j,k,l,n = 0

    cdef long new_shape[6]

    new_shape = out1.shape

    with nogil, parallel(num_threads=4):
        for i in prange(new_shape[0], schedule="dynamic"):
            for j in xrange(new_shape[1]):
                for k in xrange(new_shape[2]):
                    for l in xrange(new_shape[3]):

                        out1[i,j,k,l,0,0] = (in1[i,j,k,l,0,0]*
                                           in2[i,j,k,0,0]) + \
                                          (in1[i,j,k,l,0,1]*
                                            in2[i,j,k,1,0])

                        out1[i,j,k,l,0,1] = (in1[i,j,k,l,0,0]*
                                           in2[i,j,k,0,1]) + \
                                          (in1[i,j,k,l,0,1]*
                                            in2[i,j,k,1,1])

                        out1[i,j,k,l,1,0] = (in1[i,j,k,l,1,0]*
                                           in2[i,j,k,0,0]) + \
                                          (in1[i,j,k,l,1,1]*
                                            in2[i,j,k,1,0])

                        out1[i,j,k,l,1,1] = (in1[i,j,k,l,1,0]*
                                           in2[i,j,k,0,1]) + \
                                          (in1[i,j,k,l,1,1]*
                                            in2[i,j,k,1,1])
