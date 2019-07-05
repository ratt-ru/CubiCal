# CubiCal: a radio interferometric calibration suite
# (c) 2017 Rhodes University & Jonathan S. Kenyon
# http://github.com/ratt-ru/CubiCal
# This code is distributed under the terms of GPLv2, see LICENSE.md for details
"""
Kernels for the Jones chain machine. Functions require output arrays to be 
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
from cubical.kernels import full_complex

use_parallel = True if cubical.kernels.num_omp_threads > 1 else False
use_cache = cubical.kernels.use_cache

# Allocators are the same as for the 2x2 complex kernel.
allocate_vis_array = full_complex.allocate_vis_array
allocate_gain_array = full_complex.allocate_gain_array
allocate_flag_array = full_complex.allocate_flag_array

@jit(nopython=True, fastmath=True, parallel=use_parallel, cache=use_cache, nogil=True)
def compute_jh(jh, g, t_int, f_int):
    """
    Given J\ :sup:`H` (initially populated with the model array) and gains, computes the non-zero 
    elements of J\ :sup:`H`. J\ :sup:`H` has full time and frequency resolution - solution intervals
    are used to correctly associate the gains with the model. The result here contains the useful 
    elements of J\ :sup:`H` but does not look like the analytic solution. This function is called
    multiple times during a chain computation. Each call applies a different gain term.   

    Args:
        jh (np.complex64 or np.complex128):
            Typed memoryview of J\ :sup:`H` array with dimensions (d, m, t, f, a, a, c, c).
        g (np.complex64 or np.complex128):
            Typed memoryview of gain array with dimension (d, ti, fi, a, c, c).
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

                        jh00 = jh[d,i,t,f,aa,ab,0,0]
                        jh10 = jh[d,i,t,f,aa,ab,1,0]
                        jh01 = jh[d,i,t,f,aa,ab,0,1]
                        jh11 = jh[d,i,t,f,aa,ab,1,1]

                        g00 = g[bd,bt,bf,aa,0,0]
                        g01 = g[bd,bt,bf,aa,0,1]
                        g10 = g[bd,bt,bf,aa,1,0]
                        g11 = g[bd,bt,bf,aa,1,1]

                        jh[d,i,t,f,aa,ab,0,0] = (g00*jh00 + g01*jh10)
                        jh[d,i,t,f,aa,ab,0,1] = (g00*jh01 + g01*jh11)
                        jh[d,i,t,f,aa,ab,1,0] = (g10*jh00 + g11*jh10)
                        jh[d,i,t,f,aa,ab,1,1] = (g10*jh01 + g11*jh11)

@jit(nopython=True, fastmath=True, parallel=use_parallel, cache=use_cache, nogil=True)
def apply_left_inv_jones(jhr, ginv, t_int, f_int):
    """
    Applies the inverse of a gain array the left side of J\ :sup:`H`\R. J\ :sup:`H`\R has full time 
    and frequency resolution - solution intervals are used to correctly associate the gains with 
    the model. This is a special function which is unique to the Jones chain.    

    Args:
        jhr (np.complex64 or np.complex128):
            Typed memoryview of J\ :sup:`H`\R array with dimensions (d, t, f, a, c, c).
        ginv (np.complex64 or np.complex128):
            Typed memoryview of inverse gain array with dimensions (d, ti, fi, a, c, c).
        t_int (int):
            Number of time slots per solution interval.
        f_int (int):
            Number of frequencies per solution interval.
    """
    n_dir = jhr.shape[0]
    n_tim = jhr.shape[1]
    n_fre = jhr.shape[2]
    n_ant = jhr.shape[3]

    g_dir = ginv.shape[0]

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

                    jhr00 = jhr[d,t,f,aa,0,0]
                    jhr01 = jhr[d,t,f,aa,0,1]
                    jhr10 = jhr[d,t,f,aa,1,0]
                    jhr11 = jhr[d,t,f,aa,1,1]

                    ginv00 = ginv[bd,bt,bf,aa,0,0]
                    ginv01 = ginv[bd,bt,bf,aa,0,1]
                    ginv10 = ginv[bd,bt,bf,aa,1,0]
                    ginv11 = ginv[bd,bt,bf,aa,1,1]

                    jhr[d,t,f,aa,0,0] = (ginv00*jhr00 + ginv01*jhr10)
                    jhr[d,t,f,aa,0,1] = (ginv00*jhr01 + ginv01*jhr11)
                    jhr[d,t,f,aa,1,0] = (ginv10*jhr00 + ginv11*jhr10)
                    jhr[d,t,f,aa,1,1] = (ginv10*jhr01 + ginv11*jhr11)

@jit(nopython=True, fastmath=True, parallel=use_parallel, cache=use_cache, nogil=True)
def sum_jhr_intervals(jhr, jhrint, t_int, f_int):
    """
    Collapses J\ :sup:`H`\R to be cosistent with the solution intervals for the current gain of 
    interest. This is necessary as each gain term in a chain may have unique solution intervals. 

    Args:
        jhr (np.complex64 or np.complex128):
            Typed memoryview of J\ :sup:`H`\R array with dimensions (d, t, f, a, c, c).
        jhrint (np.complex64 or np.complex128):
            Typed memoryview of collapsed J\ :sup:`H`\R array with dimensions (d, ti, fi, a, c, c).
        t_int (int):
            Number of time slots per solution interval.
        f_int (int):
            Number of frequencies per solution interval.
    """

    n_dir = jhr.shape[0]
    n_tim = jhr.shape[1]
    n_fre = jhr.shape[2]
    n_ant = jhr.shape[3]

    broadcast_times = np.array([t//t_int for t in range(n_tim)])
    broadcast_freqs = np.array([f//f_int for f in range(n_fre)])

    for aa in prange(n_ant):
        for t in range(n_tim):
            bt = broadcast_times[t]
            for f in range(n_fre):
                bf = broadcast_freqs[f]
                for d in range(n_dir):

                    jhrint[d,bt,bf,aa,0,0] += jhr[d,t,f,aa,0,0]
                    jhrint[d,bt,bf,aa,0,1] += jhr[d,t,f,aa,0,1]
                    jhrint[d,bt,bf,aa,1,0] += jhr[d,t,f,aa,1,0]
                    jhrint[d,bt,bf,aa,1,1] += jhr[d,t,f,aa,1,1]

@jit(nopython=True, fastmath=True, parallel=use_parallel, cache=use_cache, nogil=True)
def compute_residual(m, r):
    """
    Given the model array (already multipled by the gains) and the residual array (already populated
    with the observed data), computes the final residual. This is a special case where the gains
    are applied to the model elsewhere.

    NOTE: It may be possible to accelerate this if m and r and already correctly populated in the
    lower triangle. Then we can loop over all baselines and do the subtraction everywhere instead
    of the conjugation.

    Args:
        m (np.complex64 or np.complex128):
            Typed memoryview of the model array with dimensions (d, m, t, f, a, a, c, c).
        r (np.complex64 or np.complex128):
            Typed memoryview of tje residual array with dimensions (m, t, f, a, a, c, c).
    """

    n_dir = m.shape[0]
    n_mod = m.shape[1]
    n_tim = m.shape[2]
    n_fre = m.shape[3]
    n_ant = m.shape[4]

    bls = np.array([[i,j] for i in range(n_ant) for j in range(i+1, n_ant)], dtype=np.int32)
    n_bl = bls.shape[0]

    for ibl in prange(n_bl):
        aa, ab = bls[ibl][0], bls[ibl][1]  
        for i in range(n_mod):
            for t in range(n_tim):
                for f in range(n_fre):
                    for d in range(n_dir):

                        r[i,t,f,aa,ab,0,0] -= m[d,i,t,f,aa,ab,0,0]
                        r[i,t,f,aa,ab,0,1] -= m[d,i,t,f,aa,ab,0,1]
                        r[i,t,f,aa,ab,1,0] -= m[d,i,t,f,aa,ab,1,0]
                        r[i,t,f,aa,ab,1,1] -= m[d,i,t,f,aa,ab,1,1]

                        r[i,t,f,ab,aa,0,0] = r[i,t,f,aa,ab,0,0].conjugate()
                        r[i,t,f,ab,aa,1,0] = r[i,t,f,aa,ab,0,1].conjugate()
                        r[i,t,f,ab,aa,0,1] = r[i,t,f,aa,ab,1,0].conjugate()
                        r[i,t,f,ab,aa,1,1] = r[i,t,f,aa,ab,1,1].conjugate()