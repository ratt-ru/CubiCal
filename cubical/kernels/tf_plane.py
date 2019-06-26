# CubiCal: a radio interferometric calibration suite
# (c) 2017 Rhodes University & Jonathan S. Kenyon
# http://github.com/ratt-ru/CubiCal
# This code is distributed under the terms of GPLv2, see LICENSE.md for details
"""
Cython kernels for the tf plane machine. Functions require output arrays to be
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
from cubical.kernels import phase_only

use_parallel = True if cubical.kernels.num_omp_threads > 1 else False
use_cache = cubical.kernels.use_cache

# Allocators same as for generic full kernel.
allocate_vis_array = full_complex.allocate_vis_array
allocate_gain_array = full_complex.allocate_gain_array
allocate_flag_array = full_complex.allocate_flag_array

# The exception is the parameter array which has an extra dimension.
# Defines memory layout of parameter-like arrays  (axis layout is NDxNTxNFxNAxNPxNCxNC)
_param_axis_layout = [3,1,2,0,4,5,6]   # layout is ATFDP

def allocate_param_array(shape, dtype, zeros=False):
    """Allocates a param array of the desired shape, laid out in preferred order"""
    return cubical.kernels.allocate_reordered_array(shape, dtype, _param_axis_layout, zeros=zeros)

@jit(nopython=True, fastmath=True, parallel=use_parallel, cache=use_cache, nogil=True)
def compute_jhj(tmp_jhj, jhj, ts, fs, t_int, f_int):
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

    n_dir = tmp_jhj.shape[0]
    n_tim = tmp_jhj.shape[1]
    n_fre = tmp_jhj.shape[2]
    n_ant = tmp_jhj.shape[3]

    broadcast_times = np.array([t//t_int for t in range(n_tim)])
    broadcast_freqs = np.array([f//f_int for f in range(n_fre)])

    for aa in prange(n_ant):
        for t in range(n_tim):
            bt = broadcast_times[t]
            for f in range(n_fre):
                bf = broadcast_freqs[f]
                for d in range(n_dir):
                    for c in range(2):

                        tmp_jhjcc = tmp_jhj[d,t,f,aa,c,c]

                        ff = fs[f]
                        tt = ts[t]

                        jhj[d,bt,bf,aa,0,c,c] += tmp_jhjcc
                        jhj[d,bt,bf,aa,1,c,c] += ff*tmp_jhjcc
                        jhj[d,bt,bf,aa,2,c,c] += tt*tmp_jhjcc
                        jhj[d,bt,bf,aa,3,c,c] += ff*ff*tmp_jhjcc
                        jhj[d,bt,bf,aa,4,c,c] += ff*tt*tmp_jhjcc
                        jhj[d,bt,bf,aa,5,c,c] += tt*tt*tmp_jhjcc

@jit(nopython=True, fastmath=True, parallel=use_parallel, cache=use_cache, nogil=True)
def compute_jhjinv(jhj, jhjinv, eps):
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

    n_dir = jhj.shape[0]
    n_tim = jhj.shape[1]
    n_fre = jhj.shape[2]
    n_ant = jhj.shape[3]

    for aa in prange(n_ant):
        for t in range(n_tim):
            for f in range(n_fre):
                for d in range(n_dir):
                    for c in range(2):
                        jhj0cc = jhj[d,t,f,aa,0,c,c]
                        jhj1cc = jhj[d,t,f,aa,1,c,c]
                        jhj2cc = jhj[d,t,f,aa,2,c,c]
                        jhj3cc = jhj[d,t,f,aa,3,c,c]
                        jhj4cc = jhj[d,t,f,aa,4,c,c]
                        jhj5cc = jhj[d,t,f,aa,5,c,c]

                        det =  ( jhj0cc*jhj3cc*jhj5cc +
                               2*jhj1cc*jhj2cc*jhj4cc -
                                 jhj3cc*jhj2cc*jhj2cc -
                                 jhj1cc*jhj1cc*jhj5cc -
                                 jhj4cc*jhj4cc*jhj0cc )
                        
                        if det<eps:
                            for x in range(6):
                                jhjinv[d,t,f,aa,x,c,c] = 0

                        else:

                            det = 1/det

                            jhjinv[d,t,f,aa,0,c,c] = det*(jhj3cc*jhj5cc - jhj4cc*jhj4cc)
                            jhjinv[d,t,f,aa,1,c,c] = det*(jhj4cc*jhj2cc - jhj1cc*jhj5cc)
                            jhjinv[d,t,f,aa,2,c,c] = det*(jhj1cc*jhj4cc - jhj3cc*jhj2cc)
                            jhjinv[d,t,f,aa,3,c,c] = det*(jhj5cc*jhj0cc - jhj2cc*jhj2cc)
                            jhjinv[d,t,f,aa,4,c,c] = det*(jhj1cc*jhj2cc - jhj4cc*jhj0cc)
                            jhjinv[d,t,f,aa,5,c,c] = det*(jhj3cc*jhj0cc - jhj1cc*jhj1cc)

@jit(nopython=True, fastmath=True, parallel=use_parallel, cache=use_cache, nogil=True)
def compute_jhr(tmp_jhr, jhr, ts, fs, t_int, f_int):
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

    n_dir = tmp_jhr.shape[0]
    n_tim = tmp_jhr.shape[1]
    n_fre = tmp_jhr.shape[2]
    n_ant = tmp_jhr.shape[3]

    broadcast_times = np.array([t//t_int for t in range(n_tim)])
    broadcast_freqs = np.array([f//f_int for f in range(n_fre)])

    for aa in prange(n_ant):
        for t in range(n_tim):
            bt = broadcast_times[t]
            for f in range(n_fre):
                bf = broadcast_freqs[f]
                for d in range(n_dir):
                    for c in range(2):
                        tmp_jhrcc = tmp_jhr[d,t,f,aa,c,c]

                        jhr[d,bt,bf,aa,0,c,c] +=       tmp_jhrcc
                        jhr[d,bt,bf,aa,1,c,c] += fs[f]*tmp_jhrcc
                        jhr[d,bt,bf,aa,2,c,c] += ts[t]*tmp_jhrcc

@jit(nopython=True, fastmath=True, parallel=use_parallel, cache=use_cache, nogil=True)
def compute_update(jhr, jhj, upd):
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

    n_dir = jhr.shape[0]
    n_tim = jhr.shape[1]
    n_fre = jhr.shape[2]
    n_ant = jhr.shape[3]

    for aa in prange(n_ant):
        for t in range(n_tim):
            for f in range(n_fre):
                for d in range(n_dir):
                    for c in range(2):

                        jhj0cc = jhj[d,t,f,aa,0,c,c]
                        jhj1cc = jhj[d,t,f,aa,1,c,c]
                        jhj2cc = jhj[d,t,f,aa,2,c,c]
                        jhj3cc = jhj[d,t,f,aa,3,c,c]
                        jhj4cc = jhj[d,t,f,aa,4,c,c]
                        jhj5cc = jhj[d,t,f,aa,5,c,c]

                        jhr0cc = jhr[d,t,f,aa,0,c,c]
                        jhr1cc = jhr[d,t,f,aa,1,c,c]
                        jhr2cc = jhr[d,t,f,aa,2,c,c]

                        upd[d,t,f,aa,0,c,c] = jhj0cc*jhr0cc + \
                                              jhj1cc*jhr1cc + \
                                              jhj2cc*jhr2cc

                        upd[d,t,f,aa,1,c,c] = jhj1cc*jhr0cc + \
                                              jhj3cc*jhr1cc + \
                                              jhj4cc*jhr2cc

                        upd[d,t,f,aa,2,c,c] = jhj2cc*jhr0cc + \
                                              jhj4cc*jhr1cc + \
                                              jhj5cc*jhr2cc

@jit(nopython=True, fastmath=True, parallel=use_parallel, cache=use_cache, nogil=True)
def construct_gains(param, g, ts, fs, t_int, f_int):
    """
    Given the real-valued parameters of the gains, computes the complex gains.

    Args:
        param (np.float32 or np.float64):
            Typed memoryview of parameter array with dimensions (d, ti, fi, a, b, c, c).
        g (np.complex64 or np.complex128):
            Typed memoryview of gain array with dimensions (d, t, f, a, c, c).
        fs (np.float32 or np.float64):
            Typed memoryview of channel frequencies with dimension (f).
        t_int (int):
            Number of time slots per solution interval.
        f_int (int):
            Number of frequencies per solution interval.
    """

    n_dir = g.shape[0]
    n_tim = g.shape[1]
    n_fre = g.shape[2]
    n_ant = g.shape[3]

    broadcast_times = np.array([t//t_int for t in range(n_tim)])
    broadcast_freqs = np.array([f//f_int for f in range(n_fre)])

    for aa in prange(n_ant):
        for t in range(n_tim):
            bt = broadcast_times[t]
            for f in range(n_fre):
                bf = broadcast_freqs[f]
                for d in range(n_dir):
                    for c in range(2):
                        p0cc = param[d,bt,bf,aa,0,c,c]
                        p1cc = param[d,bt,bf,aa,1,c,c]
                        p2cc = param[d,bt,bf,aa,2,c,c]

                        g[d,t,f,aa,c,c] = np.exp(1j*(p0cc + fs[f]*p1cc + ts[t]*p2cc))

# Cherry-pick other methods from standard kernels

# inner_jhj is just a phase J^H.J, with intervals of 1,1
compute_inner_jhj = lambda m,jhj1: phase_only.compute_jhj(m, jhj1, 1, 1)

# inner_jhr is just a phase J^H.R with intervals of 1,1
compute_inner_jhr = lambda jh,gh,r,jhr1: phase_only.compute_jhr(jh, gh, r, jhr1, 1, 1)

# J^H computed using diagonal gains
compute_jh = diag_complex.compute_jh

# residuals computed assuming diagonal gains
compute_residual = diag_complex.compute_residual

# corrected visibilities computed assuming diagonal gains
compute_corrected = diag_complex.compute_corrected

# gains applied as diagonal
apply_gains = diag_complex.apply_gains

# gains inverted as diagonal
invert_gains = generics.compute_diag_inverse
