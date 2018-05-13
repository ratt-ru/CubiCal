# CubiCal: a radio interferometric calibration suite
# (c) 2017 Rhodes University & Jonathan S. Kenyon
# http://github.com/ratt-ru/CubiCal
# This code is distributed under the terms of GPLv2, see LICENSE.md for details

# second type, for mixing e.g. models and gains of different precision
ctypedef fused complex3264a:
    np.complex64_t
    np.complex128_t

#
# This defines a standard set of kernel methods for diagonal phase-only gains. It is reused in the
# the cyphase_only and cydiag_phase_only kernels.
#

cygenerics = cubical.kernels.import_kernel("cygenerics")
cyfull = cubical.kernels.import_kernel("cyfull_complex")
cydiag = cubical.kernels.import_kernel("cydiag_complex")
cydiagdiag = cubical.kernels.import_kernel("cydiagdiag_complex")


allocate_vis_array = cyfull.allocate_vis_array
allocate_gain_array = cyfull.allocate_gain_array
allocate_flag_array = cyfull.allocate_flag_array
allocate_param_array = cyfull.allocate_param_array


### defines memory layout of model-like arrays (axis layout is NDxNMxNTxNFxNAxNAxNCxNC)
#_model_axis_layout = [1,2,3,0,4,5,6,7]
#_model_axis_layout = cyfull._model_axis_layout
_model_axis_layout = [4,5,1,2,3,0,6,7]

### defines memory layout of gain-like arrays  (axis layout is NDxNTxNFxNAxNCxNC)
#_gain_axis_layout = [1,2,0,3,4,5]
_gain_axis_layout = cyfull._gain_axis_layout

### defines memory layout of flag-like arrays  (axis layout is NDxNTxNFxNA)
# _flag_axis_layout = _gain_axis_layoyt[:-2]
_flag_axis_layout = cyfull._flag_axis_layout

### defines memory layout of parameter-like arrays
# _param_axis_layout = _gain_axis_layout

def allocate_vis_array(shape, dtype, zeros=False):
     """Allocates a visibility array of the desired shape, laid out in preferred order"""
     return cubical.kernels.allocate_reordered_array(shape, dtype, _model_axis_layout, zeros=zeros)

def allocate_gain_array(shape, dtype, zeros=False):
    """Allocates a gain array of the desired shape, laid out in preferred order"""
    return cubical.kernels.allocate_reordered_array(shape, dtype, _gain_axis_layout, zeros=zeros)

def allocate_flag_array(shape, dtype, zeros=False):
    """Allocates a flag array of the desired shape, laid out in preferred order"""
    return cubical.kernels.allocate_reordered_array(shape, dtype, _flag_axis_layout, zeros=zeros)

allocate_param_array = allocate_gain_array


@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def cycompute_jhj(complex3264 [:,:,:,:,:,:,:,:] m,
                  complex3264a   [:,:,:,:,:,:] jhj,
                  int t_int=1,
                  int f_int=1):

    """
    Given the model array, computes the diagonal entries of J\ :sup:`H`\J. J\ :sup:`H`\J is computed
    over intervals. This is an approximation of the Hessian. In the phase-only case, this
    approximation does not depend on the gains, therefore it does not vary with iteration.

    Args:
        m (np.complex64 or np.complex128):
            Typed memoryview of model array with dimensions (d, m, t, f, a, a, c, c).
        jhj (np.complex64 or np.complex128):
            Typed memoryview of J\ :sup:`H`\J array with dimensions (d, ti, fi, a, c, c). Must be zero-filled.
        t_int (int):
            Number of time slots per solution interval.
        f_int (int):
            Number of frequencies per solution interval.
    """

    cdef int d, i, t, f, aa, ab, rr, rc, gd = 0
    cdef int n_dir, n_mod, n_tim, n_fre, n_ant

    n_dir = m.shape[0]
    n_mod = m.shape[1]
    n_tim = m.shape[2]
    n_fre = m.shape[3]
    n_ant = m.shape[4]

    cdef int num_threads = cubical.kernels.num_omp_threads

    with nogil, parallel(num_threads=num_threads):
        for aa in prange(n_ant, schedule='static'):
            for ab in xrange(n_ant):
                for i in xrange(n_mod):
                    for t in xrange(n_tim):
                        rr = t/t_int
                        for f in xrange(n_fre):
                            rc = f/f_int
                            for d in xrange(n_dir):
                                add_mm_product(&jhj[d,rr,rc,aa,0,0],&m[d,i,t,f,aa,ab,0,0],&m[d,i,t,f,ab,aa,0,0])


cdef inline void get_indices4(int block,int *i0,int *i1,int *i2,int *i3,int n0,int n1,int n2,int n3) nogil:
    i3[0] = block%n3
    block /= n3
    i2[0] = block%n2
    block /= n2
    i1[0] = block%n1
    block /= n1
    i0[0] = block

cdef inline void get_indices5(int block,int *i0,int *i1,int *i2,int *i3,int *i4,int n0,int n1,int n2,int n3,int n4) nogil:
    i4[0] = block%n4
    block /= n4
    i3[0] = block%n3
    block /= n3
    i2[0] = block%n2
    block /= n2
    i1[0] = block%n1
    block /= n1
    i0[0] = block

@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def _bad_cycompute_residual_1(complex3264 [:,:,:,:,:,:,:,:] m,
                       complex3264 [:,:,:,:,:,:] g,
                       complex3264 [:,:,:,:,:,:] gh,
                       complex3264 [:,:,:,:,:,:,:] r,
                       int t_int,
                       int f_int):

    """
    Given the model, gains, and their conjugates, computes the residual. Residual has full time and
    frequency resolution - solution intervals are used to correctly associate the gains with the
    model.

    Args:
        m (np.complex64 or np.complex128):
            Typed memoryview of model array with dimensions (d, m, t, f, a, a, c, c).
        g (np.complex64 or np.complex128):
            Typed memoryview of gain array with dimension (d, ti, fi, a, c, c).
        gh (np.complex64 or np.complex128):
            Typed memoryview of conjugate gain array with dimensions (d, ti, fi, a, c, c).
        r (np.complex64 or np.complex128):
            Typed memoryview of residual array with dimensions (m, t, f, a, a, c, c).
        t_int (int):
            Number of time slots per solution interval.
        f_int (int):
            Number of frequencies per solution interval.
    """

    cdef int d, i, t, f, aa, ab, ibl, rr, rc = 0
    cdef int n_dir, n_mod, n_tim, n_fre, n_ant

    n_dir = m.shape[0]
    n_mod = m.shape[1]
    n_tim = m.shape[2]
    n_fre = m.shape[3]
    n_ant = m.shape[4]

    cdef int num_threads = cubical.kernels.num_omp_threads

    cdef int nrr = (n_tim-1)/t_int + 1
    cdef int nrc = (n_fre-1)/f_int + 1
    cdef int block, nblocks = nrr*nrc*n_mod*n_ant*n_ant

    with nogil, parallel(num_threads=num_threads):
        for block in prange(nblocks, schedule='static'):
            get_indices5(block,&i,&rr,&rc,&aa,&ab,n_mod,nrr,nrc,n_ant,n_ant)
            for t in xrange(rr*t_int, min((rr+1)*t_int, n_tim)):
                for f in xrange(rc*f_int, min((rc+1)*f_int, n_fre)):
                    for d in xrange(n_dir):
                        subtract_gmgh_product(&r[i,t,f,aa,ab,0,0], &g[d,rr,rc,aa,0,0], &m[d,i,t,f,aa,ab,0,0], &gh[d,rr,rc,ab,0,0])

@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def _bad_cycompute_jhj_1(complex3264 [:,:,:,:,:,:,:,:] m,
                  complex3264   [:,:,:,:,:,:] jhj,
                  int t_int=1,
                  int f_int=1):

    """
    Given the model array, computes the diagonal entries of J\ :sup:`H`\J. J\ :sup:`H`\J is computed
    over intervals. This is an approximation of the Hessian. In the phase-only case, this
    approximation does not depend on the gains, therefore it does not vary with iteration.

    Args:
        m (np.complex64 or np.complex128):
            Typed memoryview of model array with dimensions (d, m, t, f, a, a, c, c).
        jhj (np.complex64 or np.complex128):
            Typed memoryview of J\ :sup:`H`\J array with dimensions (d, ti, fi, a, c, c). Must be zero-filled.
        t_int (int):
            Number of time slots per solution interval.
        f_int (int):
            Number of frequencies per solution interval.
    """

    cdef int d, i, t, f, aa, ab, rr, rc, gd = 0
    cdef int n_dir, n_mod, n_tim, n_fre, n_ant

    n_dir = m.shape[0]
    n_mod = m.shape[1]
    n_tim = m.shape[2]
    n_fre = m.shape[3]
    n_ant = m.shape[4]

    cdef int num_threads = cubical.kernels.num_omp_threads

    cdef int nrr = (n_tim-1)/t_int + 1
    cdef int nrc = (n_fre-1)/f_int + 1
    cdef int block, nblocks = nrr*nrc*n_dir*n_ant

    with nogil, parallel(num_threads=num_threads):
        for block in prange(nblocks, schedule='static'):
            get_indices4(block,&rr,&rc,&d,&aa,nrr,nrc,n_dir,n_ant)
            for t in xrange(rr*t_int, min((rr+1)*t_int, n_tim)):
                for f in xrange(rc*f_int, min((rc+1)*f_int, n_fre)):
                    for ab in xrange(n_ant):
                        for i in xrange(n_mod):
                            add_mm_product(&jhj[d,rr,rc,aa,0,0],&m[d,i,t,f,aa,ab,0,0],&m[d,i,t,f,ab,aa,0,0])



@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def cycompute_jhr(complex3264 [:,:,:,:,:,:] gh,
                  complex3264a [:,:,:,:,:,:,:,:] jh,
                  complex3264a [:,:,:,:,:,:,:] r,
                  complex3264 [:,:,:,:,:,:] jhr,
                  int t_int=1,
                  int f_int=1):

    """
    Given the conjugate gains, J\ :sup:`H` and the residual (or observed data, in special cases),
    computes J\ :sup:`H`\R. J\ :sup:`H`\R is computed over intervals.

    Args:
        gh (np.complex64 or np.complex128):
            Typed memoryview of gain array with dimension (d, ti, fi, a, c, c).
        jh (np.complex64 or np.complex128):
            Typed memoryview of J\ :sup:`H` array with dimensions (d, m, t, f, a, a, c, c).
        r (np.complex64 or np.complex128):
            Typed memoryview of residual array with dimensions (m, t, f, a, a, c, c).
        jhr (np.complex64 or np.complex128):
            Typed memoryview of J\ :sup:`H`\R array with dimensions (d, ti, fi, a, c, c).
        t_int (int):
            Number of time slots per solution interval.
        f_int (int):
            Number of frequencies per solution interval.
    """

    cdef int d, i, t, f, aa, ab, rr, rc = 0, gd = 0
    cdef int n_dir, n_mod, n_tim, n_fre, n_ant, g_dir

    n_dir = jh.shape[0]
    n_mod = jh.shape[1]
    n_tim = jh.shape[2]
    n_fre = jh.shape[3]
    n_ant = jh.shape[4]
    g_dir = gh.shape[0]

    cdef int num_threads = cubical.kernels.num_omp_threads

    with nogil, parallel(num_threads=num_threads):
        for aa in prange(n_ant, schedule='static'):
            for ab in xrange(n_ant):
                for i in xrange(n_mod):
                    for t in xrange(n_tim):
                        rr = t/t_int
                        for f in xrange(n_fre):
                            rc = f/f_int
                            for d in xrange(n_dir):
                                gd = d%g_dir
                                add_ghrjh_product(&jhr[d,rr,rc,aa,0,0], &gh[gd,rr,rc,aa,0,0],
                                                  &r[i,t,f,aa,ab,0,0], &jh[d,i,t,f,ab,aa,0,0])

@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def cycompute_update(float3264 [:,:,:,:,:,:] jhr,
                     float3264 [:,:,:,:,:,:] jhjinv,
                     float3264 [:,:,:,:,:,:] upd):
    """
    Given J\ :sup:`H`\R and (J\ :sup:`H`\J)\ :sup:`-1`, computes the gain update. The dimensions of
    the input should already be consistent, making this operation simple. These arrays are real
    valued.

    Args:
        jhr (np.float32 or np.float64):
            Typed memoryview of J\ :sup:`H`\R array with dimensions (d, ti, fi, a, c, c).
        jhjinv (np.float32 or np.float64):
            Typed memoryview of (J\ :sup:`H`\J)\ :sup:`-1` array with dimensions
            (d, ti, fi, a, c, c).
        upd (np.float32 or np.float64):
            Typed memoryview of gain update array with dimensions (d, ti, fi, a, c, c).
    """

    cdef int d, t, f, aa = 0
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
                        upd[d,t,f,aa,0,0] = jhjinv[d,t,f,aa,0,0]*jhr[d,t,f,aa,0,0]
                        upd[d,t,f,aa,0,1] = upd[d,t,f,aa,1,0] = 0
                        upd[d,t,f,aa,1,1] = jhjinv[d,t,f,aa,1,1]*jhr[d,t,f,aa,1,1]

