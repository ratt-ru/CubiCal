# CubiCal: a radio interferometric calibration suite
# (c) 2017 Rhodes University & Jonathan S. Kenyon
# http://github.com/ratt-ru/CubiCal
# This code is distributed under the terms of GPLv2, see LICENSE.md for details
"""
Cython kernels for the time and frequency slopes (delay and rates) gain machine. Functions require 
output arrays to be provided. Common dimensions of arrays are:

+----------------+------+
| Dimension      | Size |
+================+======+
| Direction      |   d  |
+----------------+------+
| Model          |   m  |
+----------------+------+
| Time           |   t  |
+----------------+------+
| Time Intervals |   ti |
+----------------+------+
| Frequency      |   f  |
+----------------+------+
| Freq Intervals |   fi |
+----------------+------+
| Antenna        |   a  |
+----------------+------+
| Block          |   b  |
+----------------+------+
| Correlation    |   c  |
+----------------+------+

"""
import numpy as np
cimport numpy as np
import cython

ctypedef fused complex3264:
    np.complex64_t
    np.complex128_t

ctypedef fused float3264:
    np.float32_t
    np.float64_t

@cython.boundscheck(False)
cdef inline void update_jhj_element(float3264 [:,:,:,:,:,:] jhj1,
                                    float3264 [:,:,:,:,:,:,:] jhj,
                                    float3264 [:] ts,
                                    float3264 [:] fs,
                                    int d,int t,int f,int rr,int rc,int aa,int c) nogil:
    """Inner loop of compute_jhj"""
    jhj[d,rr,rc,aa,0,c,c] += jhj1[d,t,f,aa,c,c]
    jhj[d,rr,rc,aa,1,c,c] += fs[f]*jhj1[d,t,f,aa,c,c]
    jhj[d,rr,rc,aa,2,c,c] += ts[t]*jhj1[d,t,f,aa,c,c]
    jhj[d,rr,rc,aa,3,c,c] += fs[f]*fs[f]*jhj1[d,t,f,aa,c,c]
    jhj[d,rr,rc,aa,4,c,c] += fs[f]*ts[t]*jhj1[d,t,f,aa,c,c]
    jhj[d,rr,rc,aa,5,c,c] += ts[t]*ts[t]*jhj1[d,t,f,aa,c,c]


@cython.cdivision(True)
@cython.boundscheck(False)
cdef inline void compute_jhjinv_element(float3264 [:,:,:,:,:,:,:] jhj,
                                        float3264 [:,:,:,:,:,:,:] jhjinv,
                                        float eps,
                                        int d,int t,int f,int aa,int c) nogil:
    """Inner loop of compute_jhjinv"""
    cdef float3264 det =  ( jhj[d,t,f,aa,0,c,c]*jhj[d,t,f,aa,3,c,c]*jhj[d,t,f,aa,5,c,c] +
                          2*jhj[d,t,f,aa,1,c,c]*jhj[d,t,f,aa,2,c,c]*jhj[d,t,f,aa,4,c,c] -
                            jhj[d,t,f,aa,3,c,c]*jhj[d,t,f,aa,2,c,c]*jhj[d,t,f,aa,2,c,c] -
                            jhj[d,t,f,aa,1,c,c]*jhj[d,t,f,aa,1,c,c]*jhj[d,t,f,aa,5,c,c] -
                            jhj[d,t,f,aa,4,c,c]*jhj[d,t,f,aa,4,c,c]*jhj[d,t,f,aa,0,c,c] )

    if det<eps:
        for x in xrange(6):
            jhjinv[d,t,f,aa,x,c,c] = 0

    else:

        det = 1/det

        jhjinv[d,t,f,aa,0,c,c] = det*(jhj[d,t,f,aa,3,c,c]*jhj[d,t,f,aa,5,c,c] -
                                      jhj[d,t,f,aa,4,c,c]*jhj[d,t,f,aa,4,c,c])
        jhjinv[d,t,f,aa,1,c,c] = det*(jhj[d,t,f,aa,4,c,c]*jhj[d,t,f,aa,2,c,c] -
                                      jhj[d,t,f,aa,1,c,c]*jhj[d,t,f,aa,5,c,c])
        jhjinv[d,t,f,aa,2,c,c] = det*(jhj[d,t,f,aa,1,c,c]*jhj[d,t,f,aa,4,c,c] -
                                      jhj[d,t,f,aa,3,c,c]*jhj[d,t,f,aa,2,c,c])
        jhjinv[d,t,f,aa,3,c,c] = det*(jhj[d,t,f,aa,5,c,c]*jhj[d,t,f,aa,0,c,c] -
                                      jhj[d,t,f,aa,2,c,c]*jhj[d,t,f,aa,2,c,c])
        jhjinv[d,t,f,aa,4,c,c] = det*(jhj[d,t,f,aa,1,c,c]*jhj[d,t,f,aa,2,c,c] -
                                      jhj[d,t,f,aa,4,c,c]*jhj[d,t,f,aa,0,c,c])
        jhjinv[d,t,f,aa,5,c,c] = det*(jhj[d,t,f,aa,3,c,c]*jhj[d,t,f,aa,0,c,c] -
                                      jhj[d,t,f,aa,1,c,c]*jhj[d,t,f,aa,1,c,c])




@cython.boundscheck(False)
cdef inline void update_jhr_element(float3264 [:,:,:,:,:,:] jhr1,
                                    float3264 [:,:,:,:,:,:,:] jhr,
                                    float3264 [:] ts,
                                    float3264 [:] fs,
                                    int d,int t,int f,int rr,int rc,int aa,int c) nogil:
    """Inner loop of compute_jhr"""
    jhr[d,rr,rc,aa,0,c,c] +=       jhr1[d,t,f,aa,c,c]
    jhr[d,rr,rc,aa,1,c,c] += fs[f]*jhr1[d,t,f,aa,c,c]
    jhr[d,rr,rc,aa,2,c,c] += ts[t]*jhr1[d,t,f,aa,c,c]


@cython.boundscheck(False)
cdef inline void compute_update_element(float3264 [:,:,:,:,:,:,:] jhj,
                                        float3264 [:,:,:,:,:,:,:] jhr,
                                        float3264 [:,:,:,:,:,:,:] upd,
                                        int d,int t,int f,int aa,int c) nogil:
    """Inner loop of compute_update"""
    upd[d,t,f,aa,0,c,c] = jhj[d,t,f,aa,0,c,c]*jhr[d,t,f,aa,0,c,c] + \
                          jhj[d,t,f,aa,1,c,c]*jhr[d,t,f,aa,1,c,c] + \
                          jhj[d,t,f,aa,2,c,c]*jhr[d,t,f,aa,2,c,c]

    upd[d,t,f,aa,1,c,c] = jhj[d,t,f,aa,1,c,c]*jhr[d,t,f,aa,0,c,c] + \
                          jhj[d,t,f,aa,3,c,c]*jhr[d,t,f,aa,1,c,c] + \
                          jhj[d,t,f,aa,4,c,c]*jhr[d,t,f,aa,2,c,c]

    upd[d,t,f,aa,2,c,c] = jhj[d,t,f,aa,2,c,c]*jhr[d,t,f,aa,0,c,c] + \
                          jhj[d,t,f,aa,4,c,c]*jhr[d,t,f,aa,1,c,c] + \
                          jhj[d,t,f,aa,5,c,c]*jhr[d,t,f,aa,2,c,c]


cdef extern from "<complex.h>" namespace "std":
    double complex exp(double complex z)
    float complex exp(float complex z)  # overload

@cython.boundscheck(False)
cdef inline float3264 compute_phase_element(float3264 [:,:,:,:,:,:,:] param,
                                            float3264 [:] ts,
                                            float3264 [:] fs,
                                             int d,int t,int f,int rr,int rc,int aa,int c) nogil:
    """inner loop of construct_gains"""
    return param[d,rr,rc,aa,0,c,c] + fs[f]*param[d,rr,rc,aa,1,c,c] + ts[t]*param[d,rr,rc,aa,2,c,c]


include "includes/slope_kernel.pxi"

### cherry-pick other methods from standard kernels

### inner_jhj is just a phase J^H.J, with intervals of 1,1
cycompute_inner_jhj = lambda m,jhj1: cyphase.cycompute_jhj(m,jhj1,1,1)

### inner_jhr is just a phase J^H.R with intervals of 1,1
cycompute_inner_jhr = lambda jh,gh,r,jhr1: cyphase.cycompute_jhr(jh,gh,r,jhr1,1,1)

### J^H computed using diagonal gains
cycompute_jh = cydiag.cycompute_jh

### residuals computed assuming diagonal gains
cycompute_residual = cydiag.cycompute_residual

### corrected visibilities computed assuming diagonal gains
cycompute_corrected = cydiag.cycompute_corrected

### gains applied as diagonal
cyapply_gains = cydiag.cyapply_gains

### gains inverted as diagonal
cyinvert_gains = cygenerics.cycompute_diag_inverse
