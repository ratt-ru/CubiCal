from builtins import range

import numpy as np
from numba import jit, prange

import cubical.kernels

use_parallel = True if cubical.kernels.num_omp_threads > 1 else False
use_cache = cubical.kernels.use_cache

@jit(nopython=True, fastmath=True, parallel=use_parallel, cache=use_cache, nogil=True)
def compute_2x2_inverse(x, xinv, flags, eps, flagbit):
    """
    Given an array x of dimensions (d,t,i,a,2,2), computes the inverse of every 2x2 block. 
    Takes flags of shape (d,t,f,a) into account, and will flag elements if the inverse is 
    too large.

    Args:
        x (np.complex64 or np.complex128):
            Typed memoryview of X array with dimensions (d, ti, fi, a, c, c)
        xinv (np.complex64 or np.complex128):
            Typed memoryview of output inverse array with dimensions
            (d, ti, fi, a, c, c)
        flags (np.uint16_t):
            Typed memoryview of flag array with dimensions (d, t, f, a)
        eps (float):
            Threshold beneath which the denominator is regarded as too small for inversion.
        flagbit (int):
            The bitflag which will be raised if flagging is required.

    Returns:
        int:
            Number of elements flagged
    """

    flag_count = 0

    eps = eps**2

    n_dir = x.shape[0]
    n_tim = x.shape[1]
    n_fre = x.shape[2]
    n_ant = x.shape[3]

    for aa in prange(n_ant):
        for t in range(n_tim):
            for f in range(n_fre):
                for d in range(n_dir):
                    if flags[d,t,f,aa]:

                        xinv[d,t,f,aa,0,0] = 0
                        xinv[d,t,f,aa,1,1] = 0
                        xinv[d,t,f,aa,0,1] = 0
                        xinv[d,t,f,aa,1,0] = 0

                    else:
                        denom = x[d,t,f,aa,0,0] * x[d,t,f,aa,1,1] - \
                                x[d,t,f,aa,0,1] * x[d,t,f,aa,1,0]

                        if (denom*denom.conjugate()).real<=eps:
                            
                            xinv[d,t,f,aa,0,0] = 0
                            xinv[d,t,f,aa,1,1] = 0
                            xinv[d,t,f,aa,0,1] = 0
                            xinv[d,t,f,aa,1,0] = 0

                            flags[d,t,f,aa] = flagbit
                            flag_count += 1

                        else:

                            xinv[d,t,f,aa,0,0] = x[d,t,f,aa,1,1]/denom
                            xinv[d,t,f,aa,1,1] = x[d,t,f,aa,0,0]/denom
                            xinv[d,t,f,aa,0,1] = -1 * x[d,t,f,aa,0,1]/denom
                            xinv[d,t,f,aa,1,0] = -1 * x[d,t,f,aa,1,0]/denom

    return flag_count

@jit(nopython=True, fastmath=True, parallel=use_parallel, cache=use_cache, nogil=True)
def compute_diag_inverse(x, xinv, flags, eps, flagbit):
    """
    Given an array x of dimensions (d,t,i,a,2,2), computes the inverse of every 2x2 block,
    under the assumption that the off-diagonal entries are zero. Takes flags of shape 
    (d,t,f,a) into account, and will flag elements if the inverse is too large.

    Args:
        x (np.complex64 or np.complex128):
            Typed memoryview of X array with dimensions (d, ti, fi, a, c, c)
        xinv (np.complex64 or np.complex128):
            Typed memoryview of output inverse array with dimensions
            (d, ti, fi, a, c, c)
        flags (np.uint16_t):
            Typed memoryview of flag array with dimensions (d, t, f, a)
        eps (float):
            Threshold beneath which the denominator is regarded as too small for inversion.
        flagbit (int):
            The bitflag which will be raised if flagging is required.

    Returns:
        int:
            Number of elements flagged
    """

    flag_count = 0
    eps = eps**2

    n_dir = x.shape[0]
    n_tim = x.shape[1]
    n_fre = x.shape[2]
    n_ant = x.shape[3]

    for aa in prange(n_ant):
        for t in range(n_tim):
            for f in range(n_fre):
                for d in range(n_dir):
                    
                    xinv[d,t,f,aa,0,1] = xinv[d,t,f,aa,1,0] = 0
                    
                    if flags[d,t,f,aa]:
                        xinv[d,t,f,aa,0,0] = xinv[d,t,f,aa,1,1] = 0
                    
                    else:
                        denom = x[d,t,f,aa,0,0] * x[d,t,f,aa,1,1]
                        if (denom.real**2 + denom.imag**2)<=eps:
                            xinv[d,t,f,aa,0,0] = xinv[d,t,f,aa,1,1] = 0
                            flags[d,t,f,aa] = flagbit
                            flag_count += 1
                        else:
                            xinv[d,t,f,aa,0,0] = 1/x[d,t,f,aa,0,0]
                            xinv[d,t,f,aa,1,1] = 1/x[d,t,f,aa,1,1]

    return flag_count

@jit(nopython=True, fastmath=True, parallel=use_parallel, cache=use_cache, nogil=True)
def compute_chisq(r, chisq):
    """
    Compute chi-square over correlations, models, and one antenna axis.

    Args:
        r (np.complex64 or np.complex128):
            Array with dimensions (i, t, f , a, a, c, c)
        chisq (np.float64):
            Array with dimensions (t, f, a)
    """
    n_mod = r.shape[0]
    n_tim = r.shape[1]
    n_fre = r.shape[2]
    n_ant = r.shape[3]

    for aa in prange(n_ant):
        for ab in range(n_ant):
            for i in range(n_mod):
                for t in range(n_tim):
                    for f in range(n_fre):
                        for c1 in range(2):
                            for c2 in range(2):
                                chisq[t,f,aa] += r[i,t,f,aa,ab,c1,c2].real**2 + r[i,t,f,aa,ab,c1,c2].imag**2

@jit(nopython=True, fastmath=True, parallel=use_parallel, cache=use_cache, nogil=True)
def compute_chisq_diag(r, chisq):
    """
    Compute chi-square over diagonal correlations, models, and one antenna axis.

    Args:
        r (np.complex64 or np.complex128):
            Array with dimensions (i, t, f , a, a, c, c)
        chisq (np.float64):
            Array with dimensions (t, f, a)
    """
    n_mod = r.shape[0]
    n_tim = r.shape[1]
    n_fre = r.shape[2]
    n_ant = r.shape[3]

    for aa in prange(n_ant):
        for ab in range(n_ant):
            for i in range(n_mod):
                for t in range(n_tim):
                    for f in range(n_fre):
                        for c in range(2):
                            chisq[t,f,aa] += r[i,t,f,aa,ab,c,c].real**2 + r[i,t,f,aa,ab,c,c].imag**2

@jit(nopython=True, fastmath=True, parallel=use_parallel, cache=use_cache, nogil=True)
def compute_chisq_offdiag(r, chisq):
    """
    Compute chi-square over off-diagonal correlations, models, and one antenna axis.

    Args:
        r (np.complex64 or np.complex128):
            Array with dimensions (i, t, f , a, a, c, c)
        chisq (np.float64):
            Array with dimensions (t, f, a)
    """
    n_mod = r.shape[0]
    n_tim = r.shape[1]
    n_fre = r.shape[2]
    n_ant = r.shape[3]

    for aa in prange(n_ant):
        for ab in range(n_ant):
            for i in range(n_mod):
                for t in range(n_tim):
                    for f in range(n_fre):
                        for c in range(2):
                            chisq[t,f,aa] += r[i,t,f,aa,ab,c,1-c].real**2 + r[i,t,f,aa,ab,c,1-c].imag**2
