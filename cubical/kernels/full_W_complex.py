# CubiCal: a radio interferometric calibration suite
# (c) 2017 Rhodes University & Jonathan S. Kenyon
# http://github.com/ratt-ru/CubiCal
# This code is distributed under the terms of GPLv2, see LICENSE.md for details
"""
Cython kernels for the robust 2x2 complex gain machine. Functions require output 
arrays to be provided. Common dimensions of arrays are:

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

# compute_residual is identical to the general full complex case.
compute_residual = full_complex.compute_residual

# compute_jh is identical to the general full complex case.
compute_jh = full_complex.compute_jh

# compute_update is identical to the general full complex case.
compute_update = full_complex.compute_update

# compute_corrected is identical to the general full complex case.
compute_corrected = full_complex.compute_corrected

# apply_gains is identical to the general full complex case.
apply_gains = full_complex.apply_gains

# right_multiply_gains is identical to the general full complex case.
right_multiply_gains = full_complex.right_multiply_gains

# Map the J^H.J inversion method to a generic inversion.
compute_jhjinv = generics.compute_2x2_inverse

# Map inversion to generic 2x2 inverse.
invert_gains = generics.compute_2x2_inverse

@jit(nopython=True, fastmath=True, parallel=use_parallel, cache=use_cache, nogil=True)
def compute_jhwr(jh, r, w, jhwr, t_int, f_int):
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

                        w0 = w[i,t,f,aa,ab,0]

                        jhwr[d,bt,bf,aa,0,0] += (r00*jhh00 + r01*jhh10)*w0 
                        jhwr[d,bt,bf,aa,0,1] += (r00*jhh01 + r01*jhh11)*w0 
                        jhwr[d,bt,bf,aa,1,0] += (r10*jhh00 + r11*jhh10)*w0 
                        jhwr[d,bt,bf,aa,1,1] += (r10*jhh01 + r11*jhh11)*w0 

@jit(nopython=True, fastmath=True, parallel=use_parallel, cache=use_cache, nogil=True)
def compute_jhwj(jh, w, jhwj, t_int, f_int):
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

                        w0 = w[i,t,f,aa,ab,0]

                        jhwj[d,bt,bf,aa,0,0] += (jh00*j00 + jh01*j10)*w0
                        jhwj[d,bt,bf,aa,0,1] += (jh00*j01 + jh01*j11)*w0
                        jhwj[d,bt,bf,aa,1,0] += (jh10*j00 + jh11*j10)*w0
                        jhwj[d,bt,bf,aa,1,1] += (jh10*j01 + jh11*j11)*w0

@jit(nopython=True, fastmath=True, parallel=use_parallel, cache=use_cache, nogil=True)
def compute_weights(r, ic, w, v, npol):
    """
    This function updates the weights, using the expression: 
        w[i] = (v+2*npol)/(v + 2*r[i].T.cov.r[i])

    Args:
        r (np.complex64 or np.complex128):
            Typed memoryview of residual array with dimensions (m, t, f, a, a, c, c).
        ic (np.complex64 or np.complex128):
            Typed memoryview of inverse covariance array with dimensions (4,4).
        w (np.complex64 or np.complex128):
            Typed memoryview of weight array with dimensions (m, t, f, a, a, 1).
        v (float):
            Degrees of freedom of the t-distribution.
        npol (float):
            Number of polarizations (correlations) in use.            
    """
    n_mod = r.shape[0]
    n_tim = r.shape[1]
    n_fre = r.shape[2]
    n_ant = r.shape[3]

    all_bls = np.array([[i,j] for i in range(n_ant) for j in range(n_ant) if i!=j], dtype=np.int32)
    n_bl = all_bls.shape[0]
    
    for ibl in prange(n_bl):
        aa, ab = all_bls[ibl][0], all_bls[ibl][1]
        for i in range(n_mod):
            for t in range(n_tim):
                for f in range(n_fre):

                    r00 = r[i,t,f,aa,ab,0,0]
                    r01 = r[i,t,f,aa,ab,0,1]
                    r10 = r[i,t,f,aa,ab,1,0]
                    r11 = r[i,t,f,aa,ab,1,1]

                    rc00 = r[i,t,f,aa,ab,0,0].conjugate()
                    rc01 = r[i,t,f,aa,ab,0,1].conjugate()
                    rc10 = r[i,t,f,aa,ab,1,0].conjugate()
                    rc11 = r[i,t,f,aa,ab,1,1].conjugate()

                    denom = (rc00*ic[0,0] + rc01*ic[1,0] + rc10*ic[2,0] + rc11*ic[3,0])*r00 + \
                            (rc00*ic[0,1] + rc01*ic[1,1] + rc10*ic[2,1] + rc11*ic[3,1])*r01 + \
                            (rc00*ic[0,2] + rc01*ic[1,2] + rc10*ic[2,2] + rc11*ic[3,2])*r10 + \
                            (rc00*ic[0,3] + rc01*ic[1,3] + rc10*ic[2,3] + rc11*ic[3,3])*r11

                    w[i,t,f,aa,ab,0] = (v + npol)/(v + denom.real) # using LB derivation     

@jit(nopython=True, fastmath=True, parallel=use_parallel, cache=use_cache, nogil=True)
def compute_cov(r, ic, w):
    """
    This function computes the un-normlaised weighted covariance matrix of 
    the visibilities using the expression: 
        cov = r.conj()*w.r
    
    Args:
        r (np.complex64 or np.complex128):
            Typed memoryview of residual array with dimensions (m, t, f, a, a, c, c).
        ic (np.complex64 or np.complex128):
            Typed memoryview of weighted inverse covariance array with dimensions (4,4).
        w (np.complex64 or np.complex128):
            Typed memoryview of weight array with dimensions (m, t, f, a, a, 1).         
    """    
    n_mod = r.shape[0]
    n_tim = r.shape[1]
    n_fre = r.shape[2]
    n_ant = r.shape[3]

    bls = np.array([[i,j] for i in range(n_ant) for j in range(i+1, n_ant)], dtype=np.int32)
    n_bl = bls.shape[0]
    
    for ibl in prange(n_bl):
        aa, ab = bls[ibl][0], bls[ibl][1]  
        for i in range(n_mod):
            for t in range(n_tim):
                for f in range(n_fre):

                    r00 = r[i,t,f,aa,ab,0,0]
                    r01 = r[i,t,f,aa,ab,0,1]
                    r10 = r[i,t,f,aa,ab,1,0]
                    r11 = r[i,t,f,aa,ab,1,1]

                    rc00 = r[i,t,f,aa,ab,0,0].conjugate()
                    rc01 = r[i,t,f,aa,ab,0,1].conjugate()
                    rc10 = r[i,t,f,aa,ab,1,0].conjugate()
                    rc11 = r[i,t,f,aa,ab,1,1].conjugate()

                    w0 = w[i,t,f,aa,ab,0].real

                    w0r00 = w0*r00
                    w0r01 = w0*r01
                    w0r10 = w0*r10
                    w0r11 = w0*r11

                    ic[0,0] += rc00*w0r00
                    ic[0,1] += rc00*w0r01
                    ic[0,2] += rc00*w0r10
                    ic[0,3] += rc00*w0r11

                    ic[1,0] += rc01*w0r00
                    ic[1,1] += rc01*w0r01
                    ic[1,2] += rc01*w0r10
                    ic[1,3] += rc01*w0r11

                    ic[2,0] += rc10*w0r00
                    ic[2,1] += rc10*w0r01
                    ic[2,2] += rc10*w0r10
                    ic[2,3] += rc10*w0r11

                    ic[3,0] += rc11*w0r00
                    ic[3,1] += rc11*w0r01
                    ic[3,2] += rc11*w0r10
                    ic[3,3] += rc11*w0r11