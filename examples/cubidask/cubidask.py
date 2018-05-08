from functools import wraps
import timeit

import dask
import dask.array as da
from dask.diagnostics import (Profiler, ResourceProfiler, CacheProfiler,
                                ProgressBar, visualize)
from dask.array.core import getter

import numpy as np
from toolz import merge

import cubical.kernels.cyfull_complex as cyfull
import cubical.kernels

def Time(code, name, n=3):
    res = timeit.repeat(code, repeat=n, number=1)
    print "{:70}: {:.2f}ms (best of {})".format(name, min(res)*1000, n)

class DaskArrays(object):
    """Creates a set of dask test arrays for testing the cubical kernels. """


    def dask_array(self, schema, dtype, method=None):
        if method is None:
            method = 'zeros'

        dims = tuple(self._dims[c] for c in schema)
        chunks = tuple(self._chunks[c] for c in schema)

        if method == 'empty':
            return da.empty(dims, dtype=dtype, chunks=chunks)
        elif method == 'zeros':
            return da.zeros(dims, dtype=dtype, chunks=chunks)
        elif method == 'ones':
            return da.ones(dims, dtype=dtype, chunks=chunks)
        elif method == 'random':
            if isinstance(dtype, np.complex):
                re = da.random.random(size=dims, chunks=chunks).astype(dtype)
                im = da.random.random(size=dims, chunks=chunks).astype(dtype)
                return re + 1j*im
            else:
                return da.random.random(size=dims, chunks=chunks).astype(dtype)
        else:
            raise ValueError("Invalid creation method %s" % method)

    def __init__(self, nd=10, nm=1, nt=60, nf=32, na=28,
                       cd=1, cm=1, ct=1, cf=16, ca=28,
                       dtype=np.complex128):

        self._dims = {'d': nd, 'm': nm,
                      't': nt, 'f': nf,
                      'p': na, 'q': na,
                      'c': 2}
        self._chunks = {'d': cd, 'm': cm,
                      't': ct, 'f': cf,
                      'p': ca, 'q': ca,
                      'c': 2}

        self.o = self.dask_array("tfpqcc", dtype, method='empty')
        self.m = self.dask_array("dmtfpqcc", dtype, method='empty')
        self.r = self.dask_array("mtfpqcc", dtype)
        self.g = self.dask_array("dtfpcc", dtype, method='empty')
        self.f = self.dask_array("dtfp", np.uint16)
        self.gh = self.dask_array("dtfpcc", dtype)
        self.jh = da.zeros_like(self.m)

        self.jhr = da.zeros_like(self.g)
        self.jhj = da.zeros_like(self.g)
        self.jhjinv = da.zeros_like(self.g)
        self.upd = da.zeros_like(self.g)
        self.corr = da.zeros_like(self.o)


def da_compute_residual(m, g, gh, t_int, f_int):
    @wraps(cyfull.cycompute_residual)
    def _wrapper(m, g, gh, t_int, f_int):
        r = np.zeros(m[0].shape[1:], m[0].dtype)
        cyfull.cycompute_residual(m[0], g[0], gh[0], r, t_int, f_int)
        return r

    return da.core.atop(_wrapper, 'mtfpqcc',
                    m, 'dmtfpqcc',
                    g, 'dtfpcc',
                    gh, 'dtfpcc',
                    t_int=t_int,
                    f_int=f_int,
                    dtype=m.dtype)

def da_compute_jh(m, g, t_int, f_int):
    @wraps(cyfull.cycompute_jh)
    def _wrapper(m, g, t_int, f_int):
        jh = np.zeros_like(m)
        cyfull.cycompute_jh(m, g, jh, t_int, f_int)
        return jh

    return da.core.atop(_wrapper, 'dmtfpqcc',
                        m, 'dmtfpqcc',
                        g, 'dtfpcc',
                        t_int=t_int,
                        f_int=f_int,
                        dtype=m.dtype)


def da_compute_jhr(jh, r, t_int, f_int):
    @wraps(cyfull.cycompute_jh)
    def _wrapper(jh, r, t_int, f_int):
        d, m, t, f, p, q, c1, c2 = jh[0][0].shape
        jhr = np.zeros((d, t, f, p, c1, c2), dtype=r[0].dtype)
        cyfull.cycompute_jhr(jh[0][0], r[0], jhr, t_int, f_int)
        return jhr

    return da.core.atop(_wrapper, 'tfpqcc',
                        jh, 'dmtfpqcc',
                        r, 'mtfpqcc',
                        t_int=t_int,
                        f_int=f_int,
                        dtype=r.dtype)


def da_compute_jhj(jh, t_int, f_int):
    @wraps(cyfull.cycompute_jh)
    def _wrapper(jh, t_int, f_int):
        d, m, t, f, p, q, c1, c2 = jh[0][0].shape
        jhj = np.zeros((d, t, f, p, c1, c2), dtype=jh[0][0].dtype)
        cyfull.cycompute_jhj(jh[0][0], jhj, t_int, f_int)
        return jhj

    return da.core.atop(_wrapper, 'dtfpcc',
                        jh, 'dmtfpqcc',
                        t_int=t_int,
                        f_int=f_int,
                        dtype=jh.dtype)

def da_compute_jhjinv(jhj, gflags, eps, flag_bit):
    @wraps(cyfull.cycompute_jhjinv)
    def _wrapper(jhj, gflags, eps, flag_bit):
        jhjinv = np.zeros_like(jhj)
        flag_counts = cyfull.cycompute_jhjinv(jhj, jhjinv, gflags, eps, flag_bit)
        # Must have the same number of dimensions as jhjinv
        flag_counts = np.reshape(flag_counts, (1,)*len(jhjinv.shape))
        #flag_counts = np.array([flag_counts])
        return flag_counts, jhjinv

    # Here we use a slightly more lower-level dask function
    # dask.core.top which instead of producing a dask array,
    # produces a dask graph. We do this because _wrapper
    # returns a (flag_counts, jhjinv) tuple, rather than
    # a single numpy array. We therefore need to insert
    # additional getter functions into the graph to retrieve
    # the tuple elements

    # Generate a unique name for the array graph keys
    # from input dask arrays and other parameters
    token = da.core.tokenize(jhj, gflags, eps, flag_bit)
    name = '-'.join(('jhjinv-and-flagcounts', token))

    # Generate the graph of an array tensor product
    dsk = da.core.top(_wrapper, name, 'dtfpcc',
                        jhj.name, 'dtfpcc',
                        gflags.name, 'dtfp',
                        eps=eps,
                        flag_bit=flag_bit,
                        numblocks={
                            jhj.name: jhj.numblocks,
                            gflags.name: gflags.numblocks,
                        })

    # merge input array graphs
    input_dsk = merge(*(a.__dask_graph__() for a in (jhj, gflags)))

    # Create the jhjinv array
    # _wrapper returns a tuple so we call getter on element 1
    name = '-'.join(('jhjinv', token))
    jhjinv_dsk = {(name, ) + k[1:]: (getter, v, 1) for k, v in dsk.items()}
    jhjinv_dsk = merge(dsk, input_dsk, jhjinv_dsk)
    jhjinv = da.Array(jhjinv_dsk, name, jhj.chunks, dtype=jhj.dtype)


    # Create the flagcount array.
    # _wrapper returns a tuple so we call getter on element 0
    name = '-'.join(('flagcount', token))
    flagcount_dsk = {(name, ) + k[1:]: (getter, v, 0) for k, v in dsk.items()}
    flagcount_dsk = merge(dsk, input_dsk, flagcount_dsk)
    # numblocks == number of chunks in each dimension
    # each flagcount chunk will be one scalar and the
    # number of chunks equals the number of blocks in jhjinv
    chunks = tuple((1,)*d for d in jhjinv.numblocks)
    flagcounts = da.Array(flagcount_dsk, name, chunks, dtype=np.int64)

    # Return reduction of flagcounts as well as the jhjinv array
    return flagcounts.sum(), jhjinv

def compute_js(du, t_int, f_int, eps, flag_bit):
    res = da_compute_residual(du.m, du.g, du.gh, t_int, f_int)
    jh = da_compute_jh(du.m, du.g, t_int, f_int)
    jhr = da_compute_jhr(jh, res, t_int, f_int)
    jhj = da_compute_jhj(jh, t_int, f_int)
    flagcounts, jhjinv = da_compute_jhjinv(jhj, du.f, eps, 1)

    return jhr, jhjinv, flagcounts

if __name__ == "__main__":

    for nd in (1,):
        du = DaskArrays(nd=nd,nt=512,ct=32,nf=128,cf=64,na=64,ca=64)

        t_int, f_int, eps = 1.0, 1.0, 0.95

        from multiprocessing.pool import ThreadPool
        from chest import Chest

        # 8 threads, 15GB memory pool
        optkw = {
            'pool' : ThreadPool(8),
            'cache' : Chest(available_memory=15e9)
        }

        jhj, jhjinv, flagcounts = compute_js(du, 1.0, 1.0, 0.95, 1)

        # Just sum over arrays so we don't run out of memory
        results = (jhj.sum(), jhjinv.sum(), flagcounts)

        prof = Profiler()
        rprof = ResourceProfiler()
        cprof = CacheProfiler()

        with ProgressBar(), prof, rprof, cprof, da.set_options(**optkw):
            results = dask.compute(*results, rerun_local_exceptions=True)
            print results

        visualize([prof, rprof, cprof])

