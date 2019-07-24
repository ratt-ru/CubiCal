# CubiCal: a radio interferometric calibration suite
# (c) 2017 Rhodes University & Jonathan S. Kenyon
# http://github.com/ratt-ru/CubiCal
# This code is distributed under the terms of GPLv2, see LICENSE.md for details
"""
Cython kernels for the phase-only gain machine. Functions require output arrays to be 
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
from cubical.kernels import diag_complex

use_parallel = True if cubical.kernels.num_omp_threads > 1 else False
use_cache = cubical.kernels.use_cache

allocate_vis_array = full_complex.allocate_vis_array
allocate_gain_array = full_complex.allocate_gain_array
allocate_flag_array = full_complex.allocate_flag_array
allocate_param_array = full_complex.allocate_param_array

@jit(nopython=True, fastmath=True, parallel=use_parallel, cache=use_cache, nogil=True)
def compute_jhj(m, jhj, t_int=1, f_int=1):
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

    n_dir = m.shape[0]
    n_mod = m.shape[1]
    n_tim = m.shape[2]
    n_fre = m.shape[3]
    n_ant = m.shape[4]

    all_bls = np.array([[i,j] for i in range(n_ant) for j in range(n_ant) if i!=j])
    n_bl = all_bls.shape[0]

    broadcast_times = np.array([t//t_int for t in range(n_tim)])
    broadcast_freqs = np.array([f//f_int for f in range(n_fre)])

    for ibl in prange(n_bl):
        aa, ab = all_bls[ibl,0], all_bls[ibl,1]
        for i in range(n_mod):
            for t in range(n_tim):
                bt = broadcast_times[t]
                for f in range(n_fre):
                    bf = broadcast_freqs[f]
                    for d in range(n_dir):

                        m00 = m[d,i,t,f,aa,ab,0,0]
                        m01 = m[d,i,t,f,aa,ab,0,1]
                        m10 = m[d,i,t,f,aa,ab,1,0]
                        m11 = m[d,i,t,f,aa,ab,1,1]

                        jhj[d,bt,bf,aa,0,0] += (m00*m00.conjugate() + m01*m01.conjugate())
                        jhj[d,bt,bf,aa,1,1] += (m10*m10.conjugate() + m11*m11.conjugate())

@jit(nopython=True, fastmath=True, parallel=use_parallel, cache=use_cache, nogil=True)
def compute_jhr(gh, jh, r, jhr, t_int=1, f_int=1):
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

    n_dir = jh.shape[0]
    n_mod = jh.shape[1]
    n_tim = jh.shape[2]
    n_fre = jh.shape[3]
    n_ant = jh.shape[4]
    g_dir = gh.shape[0]

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

                        r00 = r[i,t,f,aa,ab,0,0]
                        r01 = r[i,t,f,aa,ab,0,1]
                        r10 = r[i,t,f,aa,ab,1,0]
                        r11 = r[i,t,f,aa,ab,1,1]

                        jhh00 = jh[d,i,t,f,ab,aa,0,0]
                        jhh01 = jh[d,i,t,f,ab,aa,0,1]
                        jhh10 = jh[d,i,t,f,ab,aa,1,0]
                        jhh11 = jh[d,i,t,f,ab,aa,1,1]

                        jhr[d,bt,bf,aa,0,0] += gh[bd,bt,bf,aa,0,0] * (r00*jhh00 + r01*jhh10)
                        jhr[d,bt,bf,aa,1,1] += gh[bd,bt,bf,aa,1,1] * (r10*jhh01 + r11*jhh11)

@jit(nopython=True, fastmath=True, parallel=use_parallel, cache=use_cache, nogil=True)
def compute_update(jhr, jhjinv, upd):
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

    n_dir = jhr.shape[0]
    n_tim = jhr.shape[1]
    n_fre = jhr.shape[2]
    n_ant = jhr.shape[3]

    for aa in prange(n_ant):
        for t in range(n_tim):
            for f in range(n_fre):
                for d in range(n_dir):
                    upd[d,t,f,aa,0,0] = jhjinv[d,t,f,aa,0,0]*jhr[d,t,f,aa,0,0]
                    upd[d,t,f,aa,0,1] = upd[d,t,f,aa,1,0] = 0
                    upd[d,t,f,aa,1,1] = jhjinv[d,t,f,aa,1,1]*jhr[d,t,f,aa,1,1]

# Remaining kernel functions are reused.

# J^H is computed assuming diagonal gains.
compute_jh = diag_complex.compute_jh

# J^H.J inverse is computed assuming diagonal blocks.
compute_jhjinv = generics.compute_diag_inverse

# Residuals computed assuming diagonal gains.
compute_residual = diag_complex.compute_residual

# Corrected visibilities computed assuming diagonal gains.
compute_corrected = diag_complex.compute_corrected

# Gains applied as diagonal.
apply_gains = diag_complex.apply_gains

# Gains inverted as diagonal.
invert_gains = generics.compute_diag_inverse
