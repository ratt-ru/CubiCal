# CubiCal: a radio interferometric calibration suite
# (c) 2017 Rhodes University & Jonathan S. Kenyon
# http://github.com/ratt-ru/CubiCal
# This code is distributed under the terms of GPLv2, see LICENSE.md for details

# second type, for mixing e.g. models and gains of different precision
ctypedef fused complex3264a:
    np.complex64_t
    np.complex128_t

#
# This defines a standard set of kernel methods for phase slopes. It is reused in the
# cyf_slope, cyt_slope and the cytf_plane kernels.
#
from cython.parallel import parallel, prange

import cubical.kernels
cygenerics = cubical.kernels.import_kernel("cygenerics")
cyfull = cubical.kernels.import_kernel("cyfull_complex")
cydiag = cubical.kernels.import_kernel("cydiag_complex")
cyphase = cubical.kernels.import_kernel("cyphase_only")

### allocators same as for generic full kernel...
allocate_vis_array = cyfull.allocate_vis_array
allocate_gain_array = cyfull.allocate_gain_array
allocate_flag_array = cyfull.allocate_flag_array

### except the parameter array has an extra dimension
# defines memory layout of gain-like arrays  (axis layout is NDxNTxNFxNAxNPxNCxNC)
_param_axis_layout = [3,1,2,0,4,5,6]   # layout is ATFDP

def allocate_param_array(shape, dtype, zeros=False):
    """Allocates a param array of the desired shape, laid out in preferred order"""
    return cubical.kernels.allocate_reordered_array(shape, dtype, _param_axis_layout, zeros=zeros)





@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def cycompute_jhj(float3264 [:,:,:,:,:,:] tmp_jhj,
                  float3264 [:,:,:,:,:,:,:] jhj,
                  float3264 [:] ts,
                  float3264 [:] fs,
                  int t_int,
                  int f_int):
    """
    Given the intermediary J\ :sup:`H`\J and channel frequencies, computes the diagonal entries of
    J\ :sup:`H`\J. J\ :sup:`H`\J is computed over intervals. This is an approximation of the
    Hessian. In the phase-only frequency slope case, this approximation does not depend on the
    gains, therefore it does not vary with iteration. The addition of the block dimension is
    required due to the block-wise diagonal nature of J\ :sup:`H`\J.

    Args:
        tmp_jhj (np.float32 or np.float64):
            Typed memoryview of intermiadiary J\ :sup:`H`\J array with dimensions
            (d, t, f, a, c, c).
        jhj (np.float32 or np.float64):
            Typed memoryview of J\ :sup:`H`\J array with dimensions (d, ti, fi, a, b, c, c).
        fs (np.float32 or np.float64):
            Typed memoryview of channel frequencies with dimension (f).
        t_int (int):
            Number of time slots per solution interval.
        f_int (int):
            Number of frequencies per solution interval.
    """

    cdef int d, i, t, f, rr, rc, aa, c
    cdef int n_dir, n_mod, n_tim, n_fre, n_ant

    n_dir = tmp_jhj.shape[0]
    n_tim = tmp_jhj.shape[1]
    n_fre = tmp_jhj.shape[2]
    n_ant = tmp_jhj.shape[3]

    cdef int num_threads = cubical.kernels.num_omp_threads

    with nogil, parallel(num_threads=num_threads):
        for aa in prange(n_ant, schedule='static'):
            for t in xrange(n_tim):
                rr = t/t_int
                for f in xrange(n_fre):
                    rc = f/f_int
                    for d in xrange(n_dir):
                        for c in xrange(2):
                            update_jhj_element(tmp_jhj,jhj,ts,fs,d,t,f,rr,rc,aa,c)


@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def cycompute_jhjinv(float3264 [:,:,:,:,:,:,:] jhj,
                     float3264 [:,:,:,:,:,:,:] jhjinv,
                     float eps):
    """
    Given J\ :sup:`H`\J (or an array with similar dimensions), computes its inverse.

    Args:
        jhj (np.float32 or np.float64):
            Typed memoryview of J\ :sup:`H`\J array with dimensions (d, ti, fi, a, b, c, c).
        jhjinv (np.float32 or np.float64):
            Typed memoryview of (J\ :sup:`H`\J)\ :sup:`-1` array with dimensions
            (d, ti, fi, a, b, c, c).
        eps (float):
            Threshold beneath which the denominator is regarded as too small for inversion.
    """

    cdef int d, t, f, c, aa = 0
    cdef int n_dir, n_tim, n_fre, n_ant

    n_dir = jhj.shape[0]
    n_tim = jhj.shape[1]
    n_fre = jhj.shape[2]
    n_ant = jhj.shape[3]

    cdef int num_threads = cubical.kernels.num_omp_threads
    with nogil, parallel(num_threads=num_threads):
        for aa in prange(n_ant, schedule='static'):
            for t in xrange(n_tim):
                for f in xrange(n_fre):
                    for d in xrange(n_dir):
                        for c in xrange(2):
                            compute_jhjinv_element(jhj,jhjinv,eps,d,t,f,aa,c)

@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def cycompute_jhr(float3264 [:,:,:,:,:,:] tmp_jhr,
                  float3264 [:,:,:,:,:,:,:] jhr,
                  float3264 [:] ts,
                  float3264 [:] fs,
                  int t_int,
                  int f_int):
    """
    Given the intermediary J\ :sup:`H`\R and channel frequencies, computes J\ :sup:`H`\R.
    J\ :sup:`H`\R is computed over intervals. The addition of the block dimension is
    required due to the structure of J\ :sup:`H`.

    Args:
        tmp_jhr (np.float32 or np.float64):
            Typed memoryview of intermediary J\ :sup:`H`\R array with dimensions
            (d, t, f, a, c, c).
        jhr (np.float32 or np.float64):
            Typed memoryview of J\ :sup:`H`\J array with dimensions (d, ti, fi, a, b, c, c).
        fs (np.float32 or np.float64):
            Typed memoryview of channel frequencies with dimension (f).
        t_int (int):
            Number of time slots per solution interval.
        f_int (int):
            Number of frequencies per solution interval.
    """

    cdef int d, i, t, f, aa, ab, c, rr, rc = 0
    cdef int n_dir, n_mod, n_tim, n_fre, n_ant

    n_dir = tmp_jhr.shape[0]
    n_tim = tmp_jhr.shape[1]
    n_fre = tmp_jhr.shape[2]
    n_ant = tmp_jhr.shape[3]

    cdef int num_threads = cubical.kernels.num_omp_threads
    with nogil, parallel(num_threads=num_threads):
        for aa in prange(n_ant, schedule='static'):
            for t in xrange(n_tim):
                rr = t/t_int
                for f in xrange(n_fre):
                    rc = f/f_int
                    for d in xrange(n_dir):
                        for c in xrange(2):
                            update_jhr_element(tmp_jhr,jhr,ts,fs,d,t,f,rr,rc,aa,c)

@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def cycompute_update(float3264 [:,:,:,:,:,:,:] jhr,
                     float3264 [:,:,:,:,:,:,:] jhj,
                     float3264 [:,:,:,:,:,:,:] upd):
    """
    Given J\ :sup:`H`\R and (J\ :sup:`H`\J)\ :sup:`-1`, computes the gain update. The dimensions of
    the input should already be consistent, making this operation simple. These arrays are real
    valued.

    Args:
        jhr (np.float32 or np.float64):
            Typed memoryview of J\ :sup:`H`\R array with dimensions (d, ti, fi, a, b, c, c).
        jhjinv (np.float32 or np.float64):
            Typed memoryview of (J\ :sup:`H`\J)\ :sup:`-1` array with dimensions
            (d, ti, fi, a, b, c, c).
        upd (np.float32 or np.float64):
            Typed memoryview of gain update array with dimensions (d, ti, fi, a, b, c, c).
    """

    cdef int d, t, f, aa, c
    cdef int n_dir, n_tim, n_fre, n_ant

    n_dir = jhr.shape[0]
    n_tim = jhr.shape[1]
    n_fre = jhr.shape[2]
    n_ant = jhr.shape[3]

    cdef int num_threads = cubical.kernels.num_omp_threads
    with nogil, parallel(num_threads=num_threads):
        for aa in prange(n_ant, schedule='static'):
            for t in xrange(n_tim):
                for f in xrange(n_fre):
                    for d in xrange(n_dir):
                        for c in xrange(2):
                            compute_update_element(jhj,jhr,upd,d,t,f,aa,c)

## this silliness is required because I couldn't get it to call C++ polar directly
## (some funny interaction with fused types)
cdef extern from "complex_ops.h":
    double complex make_polar(double x,double y) nogil
    float complex make_polar(float x,float y) nogil

@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def cyconstruct_gains(float3264 [:,:,:,:,:,:,:] param,
                       complex3264 [:,:,:,:,:,:] g,
                       float3264 [:] ts,
                       float3264 [:] fs,
                       int t_int,
                       int f_int):
    """
    Given the real-valued parameters of the gains, computes the complex gains.

    Args:
        param (np.float32 or np.float64):
            Typed memoryview of parameter array with dimensions (d, ti, fi, a, b, c, c).
        g (np.complex64 or np.complex128):
            Typed memoryview of gain array with dimensions (d, ti, fi, a, c, c).
        fs (np.float32 or np.float64):
            Typed memoryview of channel frequencies with dimension (f).
        t_int (int):
            Number of time slots per solution interval.
        f_int (int):
            Number of frequencies per solution interval.
    """
    cdef int d, t, f, aa, rr, rc, c = 0
    cdef int n_dir, n_tim, n_fre, n_ant

    n_dir = g.shape[0]
    n_tim = g.shape[1]
    n_fre = g.shape[2]
    n_ant = g.shape[3]

    cdef int num_threads = cubical.kernels.num_omp_threads
    with nogil, parallel(num_threads=num_threads):
        for aa in prange(n_ant, schedule='static'):
            for t in xrange(n_tim):
                rr = t/t_int
                for f in xrange(n_fre):
                    rc = f/f_int
                    for d in xrange(n_dir):
                        for c in xrange(2):
                            g[d,t,f,aa,c,c] = make_polar(1,compute_phase_element(param,ts,fs,d,t,f,rr,rc,aa,c))

