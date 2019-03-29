# CubiCal: a radio interferometric calibration suite
# (c) 2017 Rhodes University & Jonathan S. Kenyon
# http://github.com/ratt-ru/CubiCal
# This code is distributed under the terms of GPLv2, see LICENSE.md for details


cygenerics = cubical.kernels.import_kernel("cygenerics")


# defines memory layout of model-like arrays (axis layout is NDxNMxNTxNFxNAxNAxNCxNC)
_model_axis_layout = [4,5,1,2,3,0,6,7]    # layout is AAMTFD

# defines memory layout of gain-like arrays  (axis layout is NDxNTxNFxNAxNCxNC)
_gain_axis_layout = [3,1,2,0,4,5]   # layout is ATFD

# defines memory layout of flag-like arrays  (axis layout is NTxNFxNAxNA)
_flag_axis_layout = [2,3,0,1]      # layout is AATF

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


#
# This defines a standard set of kernel methods for 2x2 gain matrices. It is reused in the
# the cyfull_complex, cydiag_complex and cydiagdiag_complex kernels.
#




@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def cycompute_residual(complex3264 [:,:,:,:,:,:,:,:] m,
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
    cdef int n_dir, n_mod, n_tim, n_fre, n_ant, g_dir, gd

    n_dir = m.shape[0]
    n_mod = m.shape[1]
    n_tim = m.shape[2]
    n_fre = m.shape[3]
    n_ant = m.shape[4]
    g_dir = g.shape[0]

    cdef int[:,:] baselines = cygenerics.half_baselines(n_ant)
    cdef int n_bl = baselines.shape[0]
    cdef int num_threads = cubical.kernels.num_omp_threads

    with nogil, parallel(num_threads=num_threads):
        for ibl in prange(n_bl, schedule='static'):
            aa, ab = baselines[ibl][0], baselines[ibl][1]
            for i in xrange(n_mod):
                for t in xrange(n_tim):
                    rr = t/t_int
                    for f in xrange(n_fre):
                        rc = f/f_int
                        for d in xrange(n_dir):
                            gd = d%g_dir
                            subtract_gmgh_product(&r[i,t,f,aa,ab,0,0], &g[gd,rr,rc,aa,0,0], &m[d,i,t,f,aa,ab,0,0], &gh[gd,rr,rc,ab,0,0])
                        vis_mat_conjugate(&r[i,t,f,ab,aa,0,0], &r[i,t,f,aa,ab,0,0])


@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def cycompute_jh(complex3264 [:,:,:,:,:,:,:,:] m,
                 complex3264 [:,:,:,:,:,:] g,
                 complex3264 [:,:,:,:,:,:,:,:] jh,
                 int t_int,
                 int f_int):

    """
    Given the model and gains, computes the non-zero elements of J\ :sup:`H`. J\ :sup:`H` has full
    time and frequency resolution - solution intervals are used to correctly associate the gains
    with the model. The result here contains the useful elements of J\ :sup:`H` but does not look
    like the analytic solution.

    Args:
        m (np.complex64 or np.complex128):
            Typed memoryview of model array with dimensions (d, m, t, f, a, a, c, c).
        g (np.complex64 or np.complex128):
            Typed memoryview of gain array with dimension (d, ti, fi, a, c, c).
        jh (np.complex64 or np.complex128):
            Typed memoryview of J\ :sup:`H` array with dimensions (d, m, t, f, a, a, c, c).
        t_int (int):
            Number of time slots per solution interval.
        f_int (int):
            Number of frequencies per solution interval.
    """

    cdef int d, i, t, f, aa, ab, rr, rc, gd = 0
    cdef int n_dir, n_mod, n_tim, n_fre, n_ant, g_dir

    n_dir = m.shape[0]
    n_mod = m.shape[1]
    n_tim = m.shape[2]
    n_fre = m.shape[3]
    n_ant = m.shape[4]

    g_dir = g.shape[0]

    cdef int[:,:] all_baselines = cygenerics.all_baselines(n_ant)
    cdef int ibl, n_bl = all_baselines.shape[0]
    cdef int num_threads = cubical.kernels.num_omp_threads

    with nogil, parallel(num_threads=num_threads):
        for ibl in prange(n_bl, schedule='static'):
            aa, ab = all_baselines[ibl][0], all_baselines[ibl][1]
            for i in xrange(n_mod):
                for t in xrange(n_tim):
                    rr = t/t_int
                    for f in xrange(n_fre):
                        rc = f/f_int
                        for d in xrange(n_dir):
                            gd = d%g_dir
                            mat_product_gm(&jh[d,i,t,f,aa,ab,0,0], &g[gd,rr,rc,aa,0,0], &m[d,i,t,f,aa,ab,0,0])

@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def cycompute_jhwr(complex3264 [:,:,:,:,:,:,:,:] jh,
                  complex3264 [:,:,:,:,:,:,:] r,
                  complex3264 [:,:,:,:,:,:] w,
                  complex3264 [:,:,:,:,:,:] jhwr,
                  int t_int,
                  int f_int):

    """
    Given J\ :sup:`H` and the residual (or observed data, in special cases), computes J\ :sup:`H`\R.
    J\ :sup:`H`\R is computed over intervals.

    Args:
        jh (np.complex64 or np.complex128):
            Typed memoryview of J\ :sup:`H` array with dimensions (d, m, t, f, a, a, c, c).
        r (np.complex64 or np.complex128):
            Typed memoryview of residual array with dimensions (m, t, f, a, a, c, c).
        w (np.complex64 or np.complex128):
            Typed memoryview of residual array with dimensions (m, t, f, a, a, 1).
        jhwr (np.complex64 or np.complex128):
            Typed memoryview of J\ :sup:`H`\R array with dimensions (d, ti, fi, a, c, c).
        t_int (int):
            Number of time slots per solution interval.
        f_int (int):
            Number of frequencies per solution interval.
    """

    cdef int d, i, t, f, aa, ab, rr, rc = 0
    cdef int n_dir, n_mod, n_tim, n_fre, n_ant

    n_dir = jh.shape[0]
    n_mod = jh.shape[1]
    n_tim = jh.shape[2]
    n_fre = jh.shape[3]
    n_ant = jh.shape[4]

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
                                add_rwjh_product(&jhwr[d,rr,rc,aa,0,0], &r[i,t,f,aa,ab,0,0], &w[i,t,f,aa,ab,0], &jh[d,i,t,f,ab,aa,0,0])



@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def cycompute_jhwj(complex3264 [:,:,:,:,:,:,:,:] jh,
                  complex3264 [:,:,:,:,:,:] w,
                  complex3264 [:,:,:,:,:,:] jhwj,
                  int t_int,
                  int f_int):
    """
    Given J\ :sup:`H` ,computes the diagonal entries of J\ :sup:`H`\J. J\ :sup:`H`\J is computed
    over intervals. This is an approximation of the Hessian.

    Args:
        jh (np.complex64 or np.complex128):
            Typed memoryview of J\ :sup:`H` array with dimensions (d, m, t, f, a, a, c, c).
        w (np.complex64 or np.complex128):
            Typed memoryview of residual array with dimensions (m, t, f, a, a, 1).
        jhj (np.complex64 or np.complex128):
            Typed memoryview of J\ :sup:`H`\J array with dimensions (d, ti, fi, a, c, c).
        t_int (int):
            Number of time slots per solution interval.
        f_int (int):
            Number of frequencies per solution interval.
    """

    cdef int d, i, t, f, aa, ab, rr, rc = 0
    cdef int n_dir, n_mod, n_tim, n_fre, n_ant

    n_dir = jh.shape[0]
    n_mod = jh.shape[1]
    n_tim = jh.shape[2]
    n_fre = jh.shape[3]
    n_ant = jh.shape[4]

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
                                add_jhwj_product(&jhwj[d,rr,rc,aa,0,0], &w[i,t,f,aa,ab,0], &jh[d,i,t,f,ab,aa,0,0])

@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def cycompute_update(complex3264 [:,:,:,:,:,:] jhr,
                     complex3264 [:,:,:,:,:,:] jhjinv,
                     complex3264 [:,:,:,:,:,:] upd):
    """
    Given J\ :sup:`H`\R and (J\ :sup:`H`\J)\ :sup:`-1`, computes the gain update. The dimensions of
    the input should already be consistent, making this operation simple.

    Args:
        jhr (np.complex64 or np.complex128):
            Typed memoryview of J\ :sup:`H`\R array with dimensions (d, ti, fi, a, c, c).
        jhjinv (np.complex64 or np.complex128):
            Typed memoryview of (J\ :sup:`H`\J)\ :sup:`-1` array with dimensions
            (d, ti, fi, a, c, c).
        upd (np.complex64 or np.complex128):
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
                        mat_product_update(&upd[d,t,f,aa,0,0], &jhr[d,t,f,aa,0,0], &jhjinv[d,t,f,aa,0,0])



@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def cycompute_corrected(complex3264 [:,:,:,:,:,:] o,
                        complex3264 [:,:,:,:,:,:] g,
                        complex3264 [:,:,:,:,:,:] gh,
                        complex3264 [:,:,:,:,:,:] corr,
                        int t_int,
                        int f_int):

    """
    Given the observed visbilities, inverse gains, and their conjugates, computes the corrected
    visibilitites.

    Args:
        o (np.complex64 or np.complex128):
            Typed memoryview of observed visibility array with dimensions (t, f, a, a, c, c).
        g (np.complex64 or np.complex128):
            Typed memoryview of inverse gain array with dimensions (d, ti, fi, a, c, c).
        gh (np.complex64 or np.complex128):
            Typed memoryview of inverse conjugate gain array with dimensions (d, ti, fi, a, c, c).
        corr (np.complex64 or np.complex128):
            Typed memoryview of corrected data array with dimensions (t, f, a, a, c, c).
        t_int (int):
            Number of time slots per solution interval.
        f_int (int):
            Number of frequencies per solution interval.
    """

    cdef int d, t, f, aa, ab, rr, rc = 0
    cdef int n_dir, n_tim, n_fre, n_ant

    n_dir = g.shape[0]
    n_tim = o.shape[0]
    n_fre = o.shape[1]
    n_ant = o.shape[2]

    cdef int[:,:] half_baselines = cygenerics.half_baselines(n_ant)
    cdef int ibl, n_bl = half_baselines.shape[0]
    cdef int num_threads = cubical.kernels.num_omp_threads

    with nogil, parallel(num_threads=num_threads):
        for ibl in prange(n_bl, schedule='static'):
            aa, ab = half_baselines[ibl][0], half_baselines[ibl][1]
            for t in xrange(n_tim):
                rr = t/t_int
                for f in xrange(n_fre):
                    rc = f/f_int
                    gmgh_product(&corr[t,f,aa,ab,0,0], &g[0,rr,rc,aa,0,0], &o[t,f,aa,ab,0,0], &gh[0,rr,rc,ab,0,0])
                    vis_mat_conjugate(&corr[t,f,ab,aa,0,0], &corr[t,f,aa,ab,0,0])



@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def cycompute_weights(complex3264 [:,:,:,:,:,:,:] r,
                     complex3264 [:,:] cov,
                     complex3264[:,:,:,:,:,:] w,
                     float v,
                     float npol):
    """
    This function updates the weights, using the 
    expression w[i] = (v+2*npol)/(v + 2*r[i].T.cov.r[i])
    r: the reisudals
    cov : the weigted covariance matrix inverse
    w : weights
    v : v (degrees of freedom of the t-distribution)
    npol: number of polarizations really present 2 or 4
    """

    cdef int i, d, t, f, aa, ab = 0
    cdef int n_mod, n_tim, n_fre, n_ant
    


    n_mod = r.shape[0]
    n_tim = r.shape[1]
    n_fre = r.shape[2]
    n_ant = r.shape[3]

    cdef int[:,:] all_baselines = cygenerics.all_baselines(n_ant)
    cdef int ibl, n_bl = all_baselines.shape[0]
    cdef int num_threads = cubical.kernels.num_omp_threads

    
    with nogil, parallel(num_threads=num_threads):
        for ibl in prange(n_bl, schedule='static'):
            aa, ab = all_baselines[ibl][0], all_baselines[ibl][1]   
            for i in xrange(n_mod):
                for t in xrange(n_tim):
                    for f in xrange(n_fre):
                        weight_upd_product(&w[i,t,f,aa,ab,0], &r[i,t,f,aa,ab,0,0], &cov[0,0], v, npol, lb)


@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def cycompute_cov(complex3264 [:,:,:,:,:,:,:] r,
                     complex3264 [:,:] cov,
                     complex3264[:,:,:,:,:,:] w
                     ):
    """
    This function computes the un normlaised weighted covariance matrix of 
    the visibilities using the  
    expression Cov = r.conj()*w.r
    r: the reisudals
    cov : the weigted covariance matrix inverse
    w : weights
    """

    cdef int i, d, t, f, aa, ab = 0
    cdef int n_mod, n_tim, n_fre, n_ant
    
    n_mod = r.shape[0]
    n_tim = r.shape[1]
    n_fre = r.shape[2]
    n_ant = r.shape[3]

    cdef int[:,:] half_baselines = cygenerics.half_baselines(n_ant)
    cdef int ibl, n_bl = half_baselines.shape[0]
    cdef int num_threads = cubical.kernels.num_omp_threads

    
    with nogil, parallel(num_threads=num_threads):
        for ibl in prange(n_bl, schedule='static'):
            aa, ab = half_baselines[ibl][0], half_baselines[ibl][1]   
            for i in xrange(n_mod):
                for t in xrange(n_tim):
                    for f in xrange(n_fre):
                        cov_upd_product(&cov[0,0], &r[i,t,f,aa,ab,0,0], &w[i,t,f,aa,ab,0])


@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def cyapply_gains_slow(complex3264 [:,:,:,:,:,:,:,:] m,
                  complex3264 [:,:,:,:,:,:] g,
                  complex3264 [:,:,:,:,:,:] gh,
                  int t_int,
                  int f_int):

    """
    Applies the gains and their cinjugates to the model array. This operation is performed in place
    - be wary of losing the original array. The result has full time and frequency resolution -
    solution intervals are used to correctly associate the gains with the model.

    Args:
        m (np.complex64 or np.complex128):
            Typed memoryview of model visibility array with dimensions (d, m , t, f, a, a, c, c).
        g (np.complex64 or np.complex128):
            Typed memoryview of gain array with dimensions (d, ti, fi, a, c, c).
        gh (np.complex64 or np.complex128):
            Typed memoryview of conjugate gain array with dimensions (d, ti, fi, a, c, c).
        t_int (int):
            Number of time slots per solution interval.
        f_int (int):
            Number of frequencies per solution interval.

    For reasons unknown, this ordering of the loops leads to slower performance...
    """

    cdef int d, i, t, f, aa, ab, rr, rc, gd = 0
    cdef int n_dir, n_mod, n_tim, n_fre, n_ant, g_dir
    cdef complex3264 gmtmp1, gmtmp2, gmtmp3, gmtmp4

    n_dir = m.shape[0]
    n_mod = m.shape[1]
    n_tim = m.shape[2]
    n_fre = m.shape[3]
    n_ant = m.shape[4]

    g_dir = g.shape[0]

    cdef int[:,:] half_baselines = cygenerics.half_baselines(n_ant)
    cdef int ibl, n_bl = half_baselines.shape[0]
    cdef int num_threads = cubical.kernels.num_omp_threads

    with nogil, parallel(num_threads=num_threads):
        for ibl in prange(n_bl, schedule='static'):
            aa, ab = half_baselines[ibl][0], half_baselines[ibl][1]
            for i in xrange(n_mod):
                for t in xrange(n_tim):
                    rr = t/t_int
                    for f in xrange(n_fre):
                        rc = f/f_int
                        for d in xrange(n_dir):
                            gd = d%g_dir
                            inplace_gmgh_product(&g[gd,rr,rc,aa,0,0], &m[d,i,t,f,aa,ab,0,0], &gh[gd,rr,rc,ab,0,0])
                            vis_mat_conjugate(&m[d,i,t,f,ab,aa,0,0], &m[d,i,t,f,aa,ab,0,0])


@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def cyapply_gains_3(complex3264 [:,:,:,:,:,:,:,:] m,
                  complex3264 [:,:,:,:,:,:] g,
                  complex3264 [:,:,:,:,:,:] gh,
                  int t_int,
                  int f_int):

    """
    Applies the gains and their cinjugates to the model array. This operation is performed in place
    - be wary of losing the original array. The result has full time and frequency resolution -
    solution intervals are used to correctly associate the gains with the model.

    Args:
        m (np.complex64 or np.complex128):
            Typed memoryview of model visibility array with dimensions (d, m , t, f, a, a, c, c).
        g (np.complex64 or np.complex128):
            Typed memoryview of gain array with dimensions (d, ti, fi, a, c, c).
        gh (np.complex64 or np.complex128):
            Typed memoryview of conjugate gain array with dimensions (d, ti, fi, a, c, c).
        t_int (int):
            Number of time slots per solution interval.
        f_int (int):
            Number of frequencies per solution interval.

    For reasons unknown, this ordering of the loops leads to slower performance...
    """

    cdef int d, i, t, f, aa, ab, rr, rc, gd = 0
    cdef int n_dir, n_mod, n_tim, n_fre, n_ant, g_dir
    cdef complex3264 gmtmp1, gmtmp2, gmtmp3, gmtmp4

    n_dir = m.shape[0]
    n_mod = m.shape[1]
    n_tim = m.shape[2]
    n_fre = m.shape[3]
    n_ant = m.shape[4]

    g_dir = g.shape[0]

    cdef int[:,:] half_baselines = cygenerics.half_baselines(n_ant)
    cdef int ibl, n_bl = half_baselines.shape[0]
    cdef int num_threads = cubical.kernels.num_omp_threads

    with nogil, parallel(num_threads=num_threads):
        for ibl in prange(n_bl, schedule='static'):
            aa, ab = half_baselines[ibl][0], half_baselines[ibl][1]
            for i in xrange(n_mod):
                for t in xrange(n_tim):
                    rr = t/t_int
                    for f in xrange(n_fre):
                        rc = f/f_int
                        for d in xrange(n_dir):
                            gd = d%g_dir
                            inplace_left_product(&g[gd,rr,rc,aa,0,0], &m[d,i,t,f,aa,ab,0,0])
                            inplace_right_product(&m[d,i,t,f,aa,ab,0,0], &gh[gd,rr,rc,ab,0,0])
                            vis_mat_conjugate(&m[d,i,t,f,ab,aa,0,0], &m[d,i,t,f,aa,ab,0,0])

@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def cyright_multiply_gains(complex3264 [:,:,:,:,:,:] g,
                           complex3264 [:,:,:,:,:,:] g1,
                           int t_int,
                           int f_int):

    """
    Applies the gains and their cinjugates to the model array. This operation is performed in place
    - be wary of losing the original array. The result has full time and frequency resolution -
    solution intervals are used to correctly associate the gains with the model.

    Args:
        m (np.complex64 or np.complex128):
            Typed memoryview of model visibility array with dimensions (d, m , t, f, a, a, c, c).
        g (np.complex64 or np.complex128):
            Typed memoryview of gain array with dimensions (d, ti, fi, a, c, c).
        gh (np.complex64 or np.complex128):
            Typed memoryview of conjugate gain array with dimensions (d, ti, fi, a, c, c).
        t_int (int):
            Number of time slots per solution interval.
        f_int (int):
            Number of frequencies per solution interval.

    For reasons unknown, this ordering of the loops leads to slower performance...
    """

    cdef int d, i, t, f, aa, ab, rr, rc, gd = 0
    cdef int n_dir, n_tim, n_fre, n_ant, g_dir

    n_dir = g.shape[0]
    n_tim = g.shape[1]
    n_fre = g.shape[2]
    n_ant = g.shape[3]

    g_dir = g1.shape[0]

    cdef int num_threads = cubical.kernels.num_omp_threads

    with nogil, parallel(num_threads=num_threads):
        for aa in prange(n_ant, schedule='static'):
            for t in xrange(n_tim):
                rr = t/t_int
                for f in xrange(n_fre):
                    rc = f/f_int
                    for d in xrange(n_dir):
                        gd = d%g_dir
                        inplace_right_product(&g[d,t,f,aa,0,0], &g1[gd,rr,rc,aa,0,0])



@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def cyapply_gains(complex3264 [:,:,:,:,:,:,:,:] m,
                  complex3264 [:,:,:,:,:,:] g,
                  complex3264 [:,:,:,:,:,:] gh,
                  int t_int,
                  int f_int):

    """
    Applies the gains and their conjugates to the model array. This operation is performed in place
    - be wary of losing the original array. The result has full time and frequency resolution -
    solution intervals are used to correctly associate the gains with the model.

    Args:
        m (np.complex64 or np.complex128):
            Typed memoryview of model visibility array with dimensions (d, m , t, f, a, a, c, c).
        g (np.complex64 or np.complex128):
            Typed memoryview of gain array with dimensions (d, ti, fi, a, c, c).
        gh (np.complex64 or np.complex128):
            Typed memoryview of conjugate gain array with dimensions (d, ti, fi, a, c, c).
        t_int (int):
            Number of time slots per solution interval.
        f_int (int):
            Number of frequencies per solution interval.

    For reasons unknown, this ordering of the loops leads to faster performance...
    """

    cdef int d, i, t, f, aa, ab, rr, rc, gd = 0
    cdef int n_dir, n_mod, n_tim, n_fre, n_ant, g_dir
    cdef complex3264 gmtmp1, gmtmp2, gmtmp3, gmtmp4

    n_dir = m.shape[0]
    n_mod = m.shape[1]
    n_tim = m.shape[2]
    n_fre = m.shape[3]
    n_ant = m.shape[4]

    g_dir = g.shape[0]

    cdef int[:,:] half_baselines = cygenerics.half_baselines(n_ant)
    cdef int ibl, n_bl = half_baselines.shape[0]
    cdef int num_threads = cubical.kernels.num_omp_threads

    for d in xrange(n_dir):
        gd = d%g_dir
        for i in xrange(n_mod):
            for t in xrange(n_tim):
                rr = t/t_int
                for f in xrange(n_fre):
                    rc = f/f_int
                    with nogil, parallel(num_threads=num_threads):
                        for ibl in prange(n_bl, schedule='static'):
                            aa, ab = half_baselines[ibl][0], half_baselines[ibl][1]
                            inplace_gmgh_product(&g[gd,rr,rc,aa,0,0], &m[d,i,t,f,aa,ab,0,0], &gh[gd,rr,rc,ab,0,0])
                            vis_mat_conjugate(&m[d,i,t,f,ab,aa,0,0], &m[d,i,t,f,aa,ab,0,0])

@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def cyapply_gains_1(complex3264 [:,:,:,:,:,:,:,:] m,
                  complex3264 [:,:,:,:,:,:] g,
                  complex3264 [:,:,:,:,:,:] gh,
                  int t_int,
                  int f_int):

    """
    Applies the gains and their cinjugates to the model array. This operation is performed in place
    - be wary of losing the original array. The result has full time and frequency resolution -
    solution intervals are used to correctly associate the gains with the model.

    Args:
        m (np.complex64 or np.complex128):
            Typed memoryview of model visibility array with dimensions (d, m , t, f, a, a, c, c).
        g (np.complex64 or np.complex128):
            Typed memoryview of gain array with dimensions (d, ti, fi, a, c, c).
        gh (np.complex64 or np.complex128):
            Typed memoryview of conjugate gain array with dimensions (d, ti, fi, a, c, c).
        t_int (int):
            Number of time slots per solution interval.
        f_int (int):
            Number of frequencies per solution interval.

    For reasons unknown, this ordering of the loops leads to faster performance...
    """

    cdef int d, i, t, f, aa, ab, rr, rc, gd = 0
    cdef int n_dir, n_mod, n_tim, n_fre, n_ant, g_dir
    cdef complex3264 gmtmp1, gmtmp2, gmtmp3, gmtmp4

    n_dir = m.shape[0]
    n_mod = m.shape[1]
    n_tim = m.shape[2]
    n_fre = m.shape[3]
    n_ant = m.shape[4]

    g_dir = g.shape[0]

    cdef int[:,:] half_baselines = cygenerics.half_baselines(n_ant)
    cdef int ibl, n_bl = half_baselines.shape[0]
    cdef long num_threads = cubical.kernels.num_omp_threads
    cdef long nbf = n_bl*n_fre, nbtf = nbf*n_tim, nbtfm=nbtf*n_mod, nbtfmd = nbtfm*n_dir, loop, j

    with nogil, parallel(num_threads=num_threads):
        for loop in prange(nbtfmd, schedule='static'):
            ibl = loop%n_bl
            j = loop/n_bl
            f = j%n_fre
            j = j/n_fre
            t = j%n_tim
            j = j/n_tim
            i = j%n_mod
            d = j/n_mod
            rr = t/t_int
            rc = f/f_int
            gd = d%g_dir
            aa, ab = half_baselines[ibl][0], half_baselines[ibl][1]
            inplace_gmgh_product(&g[gd,rr,rc,aa,0,0], &m[d,i,t,f,aa,ab,0,0], &gh[gd,rr,rc,ab,0,0])
            vis_mat_conjugate(&m[d,i,t,f,ab,aa,0,0], &m[d,i,t,f,aa,ab,0,0])


@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def cyapply_gains_2(complex3264 [:,:,:,:,:,:,:,:] m,
                  complex3264 [:,:,:,:,:,:] g,
                  complex3264 [:,:,:,:,:,:] gh,
                  int t_int,
                  int f_int):

    """
    Applies the gains and their cinjugates to the model array. This operation is performed in place
    - be wary of losing the original array. The result has full time and frequency resolution -
    solution intervals are used to correctly associate the gains with the model.

    Args:
        m (np.complex64 or np.complex128):
            Typed memoryview of model visibility array with dimensions (d, m , t, f, a, a, c, c).
        g (np.complex64 or np.complex128):
            Typed memoryview of gain array with dimensions (d, ti, fi, a, c, c).
        gh (np.complex64 or np.complex128):
            Typed memoryview of conjugate gain array with dimensions (d, ti, fi, a, c, c).
        t_int (int):
            Number of time slots per solution interval.
        f_int (int):
            Number of frequencies per solution interval.

    For reasons unknown, this ordering of the loops leads to faster performance...
    """

    cdef int d, i, t, f, aa, ab, rr, rc, gd = 0
    cdef int n_dir, n_mod, n_tim, n_fre, n_ant, g_dir
    cdef complex3264 gmtmp1, gmtmp2, gmtmp3, gmtmp4

    n_dir = m.shape[0]
    n_mod = m.shape[1]
    n_tim = m.shape[2]
    n_fre = m.shape[3]
    n_ant = m.shape[4]

    g_dir = g.shape[0]

    cdef int[:,:] half_baselines = cygenerics.half_baselines(n_ant)
    cdef int ibl, n_bl = half_baselines.shape[0]
    cdef int num_threads = cubical.kernels.num_omp_threads

    with nogil, parallel(num_threads=num_threads):
        for ibl in prange(n_bl, schedule='static'):
            aa, ab = half_baselines[ibl][0], half_baselines[ibl][1]
            for i in xrange(n_mod):
                for t in xrange(n_tim):
                    rr = t/t_int
                    for f in xrange(n_fre):
                        rc = f/f_int
                        for d in xrange(n_dir):
                            gd = d%g_dir
                            inplace_gmgh_product(&g[gd,rr,rc,aa,0,0], &m[d,i,t,f,aa,ab,0,0], &gh[gd,rr,rc,ab,0,0])
                            vis_mat_conjugate(&m[d,i,t,f,ab,aa,0,0], &m[d,i,t,f,aa,ab,0,0])
