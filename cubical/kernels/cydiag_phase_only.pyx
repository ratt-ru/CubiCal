# CubiCal: a radio interferometric calibration suite
# (c) 2017 Rhodes University & Jonathan S. Kenyon
# http://github.com/ratt-ru/CubiCal
# This code is distributed under the terms of GPLv2, see LICENSE.md for details
"""
Cython kernels for the diagonal vis data phase-only gain machine. Functions require output arrays to be
provided. Common dimensions of arrays are:

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
| Correlation    |   c  |
+----------------+------+

"""

import numpy as np
cimport numpy as np
import cython
from cython.parallel import parallel, prange

import cubical.kernels

ctypedef fused complex3264:
    np.complex64_t
    np.complex128_t

ctypedef fused float3264:
    np.float32_t
    np.float64_t


# now, define which set of operations to use in this kernel. The idea is, the set of inlined functions below
# determine whether we're dealing with diagonal or full 2x2 matrices -- these functions will be defined
# differently for the phase_only and diag_phase_only kernels. The actual kernel methods
# themselves are written in terms of these functions -- they're pulled in from phase_kernel.pxi below

cdef inline void add_mm_product(complex3264 * out,const complex3264 *a,const complex3264 *b) nogil:
    """
    Adds a model-model product to out (used by compute_jhj)
    """
    out[0] += a[0]*b[0]
    out[3] += a[3]*b[3]

cdef inline void add_ghrjh_product(complex3264 * out,const complex3264 *gh,const complex3264 *r,const complex3264 *jh) nogil:
    """
    Adds a G^H.R.J^H product (used by compute_jhr)
    """
    out[0] += gh[0] * r[0] * jh[0]
    out[3] += gh[3] * r[3] * jh[3]

cdef inline void subtract_gmgh_product(complex3264 * out,const complex3264 *g,const complex3264 *m,const complex3264 *gh) nogil:
    """
    Subtracts a triple 2x2 matrix product: out -= G.M.G^H
    """
    out[0] -= g[0]*m[0]*gh[0]
    out[3] -= g[3]*m[3]*gh[3]


include "includes/phase_kernel.pxi"

### Other kernel functions below are picked from other kernels, since they are generic.

### J^H is computed assuming diagonal gains
cycompute_jh = cydiagdiag.cycompute_jh

### J^H.J inverse is computed assuming diagonal blocks
cycompute_jhjinv = cygenerics.cycompute_diag_inverse

### residuals computed assuming diagonal gains
cycompute_residual = cydiagdiag.cycompute_residual

### corrected visibilities computed assuming diagonal gains
cycompute_corrected = cydiagdiag.cycompute_corrected

### gains applied as diagonal
cyapply_gains = cydiagdiag.cyapply_gains
