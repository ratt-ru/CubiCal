# CubiCal: a radio interferometric calibration suite
# (c) 2017 Rhodes University & Jonathan S. Kenyon
# http://github.com/ratt-ru/CubiCal
# This code is distributed under the terms of GPLv2, see LICENSE.md for details
"""
Cython kernels for the 2x2 complex gain machine. Functions require output arrays to be
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
from builtins import range

import numpy as np
from numba import jit, prange

import cubical.kernels
from cubical.kernels import generics

use_parallel = True if cubical.kernels.num_omp_threads > 1 else False
use_cache = cubical.kernels.use_cache

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

@jit(nopython=True, fastmath=True, parallel=use_parallel, cache=use_cache, nogil=True)
def compute_residual(m, g, gh, r, t_int, f_int):
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
    n_dir = m.shape[0]
    n_mod = m.shape[1]
    n_tim = m.shape[2]
    n_fre = m.shape[3]
    n_ant = m.shape[4]
    g_dir = g.shape[0]

    bls = np.array([[i,j] for i in range(n_ant) for j in range(i+1, n_ant)], dtype=np.int32)
    n_bl = bls.shape[0]

    broadcast_times = np.array([t//t_int for t in range(n_tim)])
    broadcast_freqs = np.array([f//f_int for f in range(n_fre)])
    broadcast_dirs = np.array([d%g_dir for d in range(n_dir)])

    for ibl in prange(n_bl):
        aa, ab = bls[ibl][0], bls[ibl][1]  
        for i in range(n_mod):
            for t in range(n_tim):
                bt = broadcast_times[t]
                for f in range(n_fre):
                    bf = broadcast_freqs[f]
                    for d in range(n_dir):
                        bd = broadcast_dirs[d]
                    
                        g00 = g[bd,bt,bf,aa,0,0]
                        g01 = g[bd,bt,bf,aa,0,1]
                        g10 = g[bd,bt,bf,aa,1,0]
                        g11 = g[bd,bt,bf,aa,1,1]

                        m00 = m[d,i,t,f,aa,ab,0,0]
                        m01 = m[d,i,t,f,aa,ab,0,1]
                        m10 = m[d,i,t,f,aa,ab,1,0]
                        m11 = m[d,i,t,f,aa,ab,1,1]

                        gh00 = gh[bd,bt,bf,ab,0,0]
                        gh01 = gh[bd,bt,bf,ab,0,1]
                        gh10 = gh[bd,bt,bf,ab,1,0]
                        gh11 = gh[bd,bt,bf,ab,1,1]

                        r[i,t,f,aa,ab,0,0] -= ((g00*m00 + g01*m10)*gh00 + (g00*m01 + g01*m11)*gh10)
                        r[i,t,f,aa,ab,0,1] -= ((g00*m00 + g01*m10)*gh01 + (g00*m01 + g01*m11)*gh11)
                        r[i,t,f,aa,ab,1,0] -= ((g10*m00 + g11*m10)*gh00 + (g10*m01 + g11*m11)*gh10)
                        r[i,t,f,aa,ab,1,1] -= ((g10*m00 + g11*m10)*gh01 + (g10*m01 + g11*m11)*gh11)

                    r[i,t,f,ab,aa,0,0] = r[i,t,f,aa,ab,0,0].conjugate()
                    r[i,t,f,ab,aa,1,0] = r[i,t,f,aa,ab,0,1].conjugate()
                    r[i,t,f,ab,aa,0,1] = r[i,t,f,aa,ab,1,0].conjugate()
                    r[i,t,f,ab,aa,1,1] = r[i,t,f,aa,ab,1,1].conjugate()     

@jit(nopython=True, fastmath=True, parallel=use_parallel, cache=use_cache, nogil=True)
def compute_jh(m, g, jh, t_int, f_int):
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
    n_dir = m.shape[0]
    n_mod = m.shape[1]
    n_tim = m.shape[2]
    n_fre = m.shape[3]
    n_ant = m.shape[4]

    g_dir = g.shape[0]

    all_bls = np.array([[i,j] for i in range(n_ant) for j in range(n_ant) if i!=j])
    n_bl = all_bls.shape[0]

    broadcast_times = np.array([t//t_int for t in range(n_tim)])
    broadcast_freqs = np.array([f//f_int for f in range(n_fre)])
    broadcast_dirs = np.array([d%g_dir for d in range(n_dir)])

    for ibl in prange(n_bl):
        aa, ab = all_bls[ibl,0], all_bls[ibl,1]
        for i in range(n_mod):
            for t in range(n_tim):
                bt = broadcast_times[t]
                for f in range(n_fre):
                    bf = broadcast_freqs[f]
                    for d in range(n_dir):
                        bd = broadcast_dirs[d]

                        g00 = g[bd,bt,bf,aa,0,0]
                        g01 = g[bd,bt,bf,aa,0,1]
                        g10 = g[bd,bt,bf,aa,1,0]
                        g11 = g[bd,bt,bf,aa,1,1]

                        m00 = m[d,i,t,f,aa,ab,0,0]
                        m01 = m[d,i,t,f,aa,ab,0,1]
                        m10 = m[d,i,t,f,aa,ab,1,0]
                        m11 = m[d,i,t,f,aa,ab,1,1]
                        
                        jh[d,i,t,f,aa,ab,0,0] = (g00*m00 + g01*m10) 
                        jh[d,i,t,f,aa,ab,0,1] = (g00*m01 + g01*m11) 
                        jh[d,i,t,f,aa,ab,1,0] = (g10*m00 + g11*m10)
                        jh[d,i,t,f,aa,ab,1,1] = (g10*m01 + g11*m11)

@jit(nopython=True, fastmath=True, parallel=use_parallel, cache=use_cache, nogil=True)
def compute_jhr(jh, r, jhr, t_int, f_int):
    """
    Given J\ :sup:`H` and the residual (or observed data, in special cases), computes J\ :sup:`H`\R.
    J\ :sup:`H`\R is computed over intervals.

    Args:
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

    n_dir = jh.shape[0]
    n_mod = jh.shape[1]
    n_tim = jh.shape[2]
    n_fre = jh.shape[3]
    n_ant = jh.shape[4]

    all_bls = np.array([[i,j] for i in range(n_ant) for j in range(n_ant) if i!=j], dtype=np.int32)
    n_bl = all_bls.shape[0]

    broadcast_times = np.array([t//t_int for t in range(n_tim)])
    broadcast_freqs = np.array([f//f_int for f in range(n_fre)])

    for ibl in prange(n_bl):
        aa, ab = all_bls[ibl][0], all_bls[ibl][1]
        for i in range(n_mod):
            for t in range(n_tim):
                bt = broadcast_times[t]
                for f in range(n_fre):
                    bf = broadcast_freqs[f]
                    for d in range(n_dir):
                        
                        r00 = r[i,t,f,aa,ab,0,0]
                        r01 = r[i,t,f,aa,ab,0,1]
                        r10 = r[i,t,f,aa,ab,1,0]
                        r11 = r[i,t,f,aa,ab,1,1]

                        jhh00 = jh[d,i,t,f,ab,aa,0,0]
                        jhh01 = jh[d,i,t,f,ab,aa,0,1]
                        jhh10 = jh[d,i,t,f,ab,aa,1,0]
                        jhh11 = jh[d,i,t,f,ab,aa,1,1]

                        jhr[d,bt,bf,aa,0,0] += (r00*jhh00 + r01*jhh10) 
                        jhr[d,bt,bf,aa,0,1] += (r00*jhh01 + r01*jhh11)
                        jhr[d,bt,bf,aa,1,0] += (r10*jhh00 + r11*jhh10)
                        jhr[d,bt,bf,aa,1,1] += (r10*jhh01 + r11*jhh11) 

@jit(nopython=True, fastmath=True, parallel=use_parallel, cache=use_cache, nogil=True)
def compute_jhj(jh, jhj, t_int, f_int):
    """
    Given J\ :sup:`H` ,computes the diagonal entries of J\ :sup:`H`\J. J\ :sup:`H`\J is computed
    over intervals. This is an approximation of the Hessian.

    Args:
        jh (np.complex64 or np.complex128):
            Typed memoryview of J\ :sup:`H` array with dimensions (d, m, t, f, a, a, c, c).
        jhj (np.complex64 or np.complex128):
            Typed memoryview of J\ :sup:`H`\J array with dimensions (d, ti, fi, a, c, c).
        t_int (int):
            Number of time slots per solution interval.
        f_int (int):
            Number of frequencies per solution interval.
    """

    n_dir = jh.shape[0]
    n_mod = jh.shape[1]
    n_tim = jh.shape[2]
    n_fre = jh.shape[3]
    n_ant = jh.shape[4]

    all_bls = np.array([[i,j] for i in range(n_ant) for j in range(n_ant) if i!=j], dtype=np.int32)
    n_bl = all_bls.shape[0]

    broadcast_times = np.array([t//t_int for t in range(n_tim)])
    broadcast_freqs = np.array([f//f_int for f in range(n_fre)])

    for ibl in prange(n_bl):
        aa, ab = all_bls[ibl][0], all_bls[ibl][1]
        for i in range(n_mod):
            for t in range(n_tim):
                bt = broadcast_times[t]
                for f in range(n_fre):
                    bf = broadcast_freqs[f]
                    for d in range(n_dir):

                        j00 = jh[d,i,t,f,ab,aa,0,0]
                        j01 = jh[d,i,t,f,ab,aa,0,1]
                        j10 = jh[d,i,t,f,ab,aa,1,0] 
                        j11 = jh[d,i,t,f,ab,aa,1,1]

                        jh00 = jh[d,i,t,f,ab,aa,0,0].conjugate()
                        jh01 = jh[d,i,t,f,ab,aa,1,0].conjugate()
                        jh10 = jh[d,i,t,f,ab,aa,0,1].conjugate() 
                        jh11 = jh[d,i,t,f,ab,aa,1,1].conjugate()

                        jhj[d,bt,bf,aa,0,0] += (jh00*j00 + jh01*j10)
                        jhj[d,bt,bf,aa,0,1] += (jh00*j01 + jh01*j11)
                        jhj[d,bt,bf,aa,1,0] += (jh10*j00 + jh11*j10)
                        jhj[d,bt,bf,aa,1,1] += (jh10*j01 + jh11*j11)

@jit(nopython=True, fastmath=True, parallel=use_parallel, cache=use_cache, nogil=True)
def compute_update(jhr, jhjinv, upd):
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

    n_dir = jhr.shape[0]
    n_tim = jhr.shape[1]
    n_fre = jhr.shape[2]
    n_ant = jhr.shape[3]

    for aa in prange(n_ant):
        for t in range(n_tim):
            for f in range(n_fre):
                for d in range(n_dir):

                    jhr00 = jhr[d,t,f,aa,0,0]
                    jhr01 = jhr[d,t,f,aa,0,1]
                    jhr10 = jhr[d,t,f,aa,1,0]
                    jhr11 = jhr[d,t,f,aa,1,1]

                    jhjinv00 = jhjinv[d,t,f,aa,0,0]
                    jhjinv01 = jhjinv[d,t,f,aa,0,1]
                    jhjinv10 = jhjinv[d,t,f,aa,1,0]
                    jhjinv11 = jhjinv[d,t,f,aa,1,1]
                    
                    upd[d,t,f,aa,0,0] = (jhr00*jhjinv00 + jhr01*jhjinv10) 
                    upd[d,t,f,aa,0,1] = (jhr00*jhjinv01 + jhr01*jhjinv11) 
                    upd[d,t,f,aa,1,0] = (jhr10*jhjinv00 + jhr11*jhjinv10)
                    upd[d,t,f,aa,1,1] = (jhr10*jhjinv01 + jhr11*jhjinv11)

@jit(nopython=True, fastmath=True, parallel=use_parallel, cache=use_cache, nogil=True)
def compute_corrected(o, g, gh, corr, t_int, f_int):
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

    n_dir = g.shape[0]
    n_tim = o.shape[0]
    n_fre = o.shape[1]
    n_ant = o.shape[2]

    bls = np.array([[i,j] for i in range(n_ant) for j in range(i, n_ant)], dtype=np.int32)
    n_bl = bls.shape[0]

    broadcast_times = np.array([t//t_int for t in range(n_tim)])
    broadcast_freqs = np.array([f//f_int for f in range(n_fre)])

    for ibl in prange(n_bl):
        aa, ab = bls[ibl][0], bls[ibl][1]
        for t in range(n_tim):
            bt = broadcast_times[t]
            for f in range(n_fre):
                bf = broadcast_freqs[f]

                g00 = g[0,bt,bf,aa,0,0]
                g01 = g[0,bt,bf,aa,0,1]
                g10 = g[0,bt,bf,aa,1,0]
                g11 = g[0,bt,bf,aa,1,1]

                o00 = o[t,f,aa,ab,0,0]
                o01 = o[t,f,aa,ab,0,1]
                o10 = o[t,f,aa,ab,1,0]
                o11 = o[t,f,aa,ab,1,1]

                gh00 = gh[0,bt,bf,ab,0,0]
                gh01 = gh[0,bt,bf,ab,0,1]
                gh10 = gh[0,bt,bf,ab,1,0]
                gh11 = gh[0,bt,bf,ab,1,1]

                corr[t,f,aa,ab,0,0] = ((g00*o00 + g01*o10)*gh00 + (g00*o01 + g01*o11)*gh10)
                corr[t,f,aa,ab,0,1] = ((g00*o00 + g01*o10)*gh01 + (g00*o01 + g01*o11)*gh11)
                corr[t,f,aa,ab,1,0] = ((g10*o00 + g11*o10)*gh00 + (g10*o01 + g11*o11)*gh10)
                corr[t,f,aa,ab,1,1] = ((g10*o00 + g11*o10)*gh01 + (g10*o01 + g11*o11)*gh11)

                corr[t,f,ab,aa,0,0] = corr[t,f,aa,ab,0,0].conjugate()
                corr[t,f,ab,aa,1,0] = corr[t,f,aa,ab,0,1].conjugate()
                corr[t,f,ab,aa,0,1] = corr[t,f,aa,ab,1,0].conjugate()
                corr[t,f,ab,aa,1,1] = corr[t,f,aa,ab,1,1].conjugate()

@jit(nopython=True, fastmath=True, parallel=use_parallel, cache=use_cache, nogil=True)
def apply_gains(m, g, gh, t_int, f_int):
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
    """

    n_dir = m.shape[0]
    n_mod = m.shape[1]
    n_tim = m.shape[2]
    n_fre = m.shape[3]
    n_ant = m.shape[4]

    g_dir = g.shape[0]

    bls = np.array([[i,j] for i in range(n_ant) for j in range(i+1, n_ant)], dtype=np.int32)
    n_bl = bls.shape[0]

    broadcast_times = np.array([t//t_int for t in range(n_tim)])
    broadcast_freqs = np.array([f//f_int for f in range(n_fre)])
    broadcast_dirs = np.array([d%g_dir for d in range(n_dir)])

    for ibl in prange(n_bl):
        aa, ab = bls[ibl][0], bls[ibl][1]  
        for i in range(n_mod):
            for t in range(n_tim):
                bt = broadcast_times[t]
                for f in range(n_fre):
                    bf = broadcast_freqs[f]
                    for d in range(n_dir):
                        bd = broadcast_dirs[d]
                        
                        g00 = g[bd,bt,bf,aa,0,0]
                        g01 = g[bd,bt,bf,aa,0,1]
                        g10 = g[bd,bt,bf,aa,1,0]
                        g11 = g[bd,bt,bf,aa,1,1]

                        m00 = m[d,i,t,f,aa,ab,0,0]
                        m01 = m[d,i,t,f,aa,ab,0,1]
                        m10 = m[d,i,t,f,aa,ab,1,0]
                        m11 = m[d,i,t,f,aa,ab,1,1]

                        gh00 = gh[bd,bt,bf,ab,0,0]
                        gh01 = gh[bd,bt,bf,ab,0,1]
                        gh10 = gh[bd,bt,bf,ab,1,0]
                        gh11 = gh[bd,bt,bf,ab,1,1]

                        tmp00 = ((g00*m00 + g01*m10)*gh00 + (g00*m01 + g01*m11)*gh10)
                        tmp01 = ((g00*m00 + g01*m10)*gh01 + (g00*m01 + g01*m11)*gh11)
                        tmp10 = ((g10*m00 + g11*m10)*gh00 + (g10*m01 + g11*m11)*gh10)
                        tmp11 = ((g10*m00 + g11*m10)*gh01 + (g10*m01 + g11*m11)*gh11)

                        m[d,i,t,f,aa,ab,0,0] = tmp00
                        m[d,i,t,f,aa,ab,0,1] = tmp01
                        m[d,i,t,f,aa,ab,1,0] = tmp10
                        m[d,i,t,f,aa,ab,1,1] = tmp11

                        m[d,i,t,f,ab,aa,0,0] = tmp00.conjugate()
                        m[d,i,t,f,ab,aa,1,0] = tmp01.conjugate()
                        m[d,i,t,f,ab,aa,0,1] = tmp10.conjugate()
                        m[d,i,t,f,ab,aa,1,1] = tmp11.conjugate()

@jit(nopython=True, fastmath=True, parallel=use_parallel, cache=use_cache, nogil=True)
def right_multiply_gains(g, g_next, t_int, f_int):
    """
    Multiples two gain terms in place. Result has full time and frequency resolution 
    even if g_next does does not. For use in Jones chain.

    NOTE: THIS MAY BE INCORRECT/BACKWARDS.

    Args:
        g (np.complex64 or np.complex128):
            Typed memoryview of gain array with dimensions (d, t, f, a, c, c).
        g_next (np.complex64 or np.complex128):
            Typed memoryview of gain array with dimensions (d, ti, fi, a, c, c).
        t_int (int):
            Number of time slots per solution interval.
        f_int (int):
            Number of frequencies per solution interval.
    """

    n_dir = g.shape[0]
    n_tim = g.shape[1]
    n_fre = g.shape[2]
    n_ant = g.shape[3]

    g_dir = g_next.shape[0]

    broadcast_times = np.array([t//t_int for t in range(n_tim)])
    broadcast_freqs = np.array([f//f_int for f in range(n_fre)])
    broadcast_dirs = np.array([d%g_dir for d in range(n_dir)])

    for aa in prange(n_ant):
        for t in range(n_tim):
            bt = broadcast_times[t]
            for f in range(n_fre):
                bf = broadcast_freqs[f]
                for d in range(n_dir):
                    bd = broadcast_dirs[d]

                    g00 = g[d,t,f,aa,0,0]
                    g01 = g[d,t,f,aa,0,1]
                    g10 = g[d,t,f,aa,1,0]
                    g11 = g[d,t,f,aa,1,1]

                    g_next00 = g_next[bd,bt,bf,aa,0,0]
                    g_next01 = g_next[bd,bt,bf,aa,0,1]
                    g_next10 = g_next[bd,bt,bf,aa,1,0]
                    g_next11 = g_next[bd,bt,bf,aa,1,1]

                    g[d,t,f,aa,0,0] = (g00*g_next00 + g01*g_next10)
                    g[d,t,f,aa,0,1] = (g00*g_next01 + g01*g_next11)
                    g[d,t,f,aa,1,0] = (g10*g_next00 + g11*g_next10)
                    g[d,t,f,aa,1,1] = (g10*g_next01 + g11*g_next11)

# Map the J^H.J inversion method to a generic inversion.
compute_jhjinv = generics.compute_2x2_inverse

# Map inversion to generic 2x2 inverse.
invert_gains = generics.compute_2x2_inverse
