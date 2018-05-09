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
from cubical.flagging import FL
import zarr

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

        self.o = self.dask_array("mtfpqcc", dtype, method='empty')
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


def compute_residual(m, g, gh, t_int, f_int):
    @wraps(cyfull.cycompute_residual)
    def _wrapper(m, g, gh, t_int, f_int):
        # Arguments to the the wrapper may be passed as numpy arrays or lists.
        # They will be a nested list of numpy array when there are
        # dimension(s) in the argument absent in the result,
        # and the nesting will equal the number of absent dimensions.
        # For example, the final result has dimensions ``mtfpqcc``
        # while argument ``m`` has dimensions ``dmtfpqcc``. Dimensions ``d``
        # is absent in the result, so ``m`` is a list with a single
        # numpy array. See
        # http://dask.pydata.org/en/latest/array-api.html#dask.array.core.atop
        # for further details
        r = np.zeros(m[0].shape[1:], m[0].dtype)
        cyfull.cycompute_residual(m[0], g[0], gh[0], r, t_int, f_int)
        return r

    # Array tensor product. Produces a dask array of supplied dimension schema
    # 'mtfpqcc' where each chunk is defined as a ``_wrapper`` call.
    # Chunks of the input dask arrays (m,g,gh) are matched to the output array
    # chunks based on their schema's ('dmtfpqcc' in the case of ``m`` for e.g.)
    # Finally, keyword arguments (t_int, f_int) are passed to
    # all ``_wrapper`` calls. atop
    # http://dask.pydata.org/en/latest/array-api.html#dask.array.core.atop
    # takes it's own keywork arguments too. Here, dtype specifies the type
    # of the output dask array.
    return da.core.atop(_wrapper, 'mtfpqcc',
                    m, 'dmtfpqcc',
                    g, 'dtfpcc',
                    gh, 'dtfpcc',
                    t_int=t_int,
                    f_int=f_int,
                    dtype=m.dtype)

def compute_jh(m, g, t_int, f_int):
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


def compute_jhr(jh, r, t_int, f_int):
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


def compute_jhj(jh, t_int, f_int):
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

def compute_jhjinv(jhj, gflags, eps, flag_bit):
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

def compute_update(jhr, jhjinv):

    @wraps(cyfull.cycompute_update)
    def _wrapper(jhr, jhjinv):
        update = np.empty_like(jhr[0])
        cyfull.cycompute_update(jhr[0], jhjinv, update)
        return update

    return da.core.atop(_wrapper, 'dtfpcc',
                        jhr, 'tfpqcc',
                        jhjinv, 'dtfpcc',
                        dtype=jhr.dtype)

def compute_js(du, t_int, f_int, eps, flag_bit):
    res = compute_residual(du.m, du.g, du.gh, t_int, f_int)
    jh = compute_jh(du.m, du.g, t_int, f_int)
    jhr = compute_jhr(jh, res, t_int, f_int)
    jhj = compute_jhj(jh, t_int, f_int)
    flagcounts, jhjinv = compute_jhjinv(jhj, du.f, eps, 1)

    return jhr, jhjinv, flagcounts




def gain_update_loop(obs_arr, model_arr, gains, gflags, t_int, f_int, eps):
    """
    """

    def _compute_gains(obs_arr, model_arr, gains, gflags, t_int, f_int, eps):
        # Extract actual ndarrays from the supplied lists
        obs_arr = obs_arr[0][0]
        model_arr = model_arr[0][0]
        gains = gains
        gflags = gflags

        loops = np.random.randint(25, 35)

        n_dir, n_tim, n_fre, n_ant, n_cor, n_cor = gains.shape

        for i in range(loops):
            jh = np.zeros_like(model_arr)
            cyfull.cycompute_jh(model_arr, gains, jh, t_int, f_int)

            jhr_shape = [n_dir, n_tim, n_fre, n_ant, n_cor, n_cor]

            jhr = np.zeros(jhr_shape, dtype=obs_arr.dtype)

            # TODO: This breaks with the new compute
            # residual code for n_dir > 1. Will need a fix.
            if n_dir > 1:
                gains_h = gains.transpose(0,1,2,3,5,4).conj()
                r = np.empty_like(obs_arr)
                cyfull.cycompute_residual(model_arr, gains, gains_h, r, t_int, f_int)
            else:
                r = obs_arr

            cyfull.cycompute_jhr(jh, r, jhr, t_int, f_int)

            jhj = np.zeros(jhr_shape, dtype=obs_arr.dtype)
            cyfull.cycompute_jhj(jh, jhj, t_int, f_int)

            jhjinv = np.empty(jhr_shape, dtype=obs_arr.dtype)
            flag_count = cyfull.cycompute_jhjinv(jhj, jhjinv, gflags, eps, FL.ILLCOND)

            update = np.empty_like(jhr)
            cyfull.cycompute_update(jhr, jhjinv, update)

            gains = update

        return gains

    return da.core.atop(_compute_gains, 'dtfpcc',
                    obs_arr, 'mtfpqcc',
                    model_arr, 'dmtfpqcc',
                    gains, 'dtfpcc',
                    gflags, 'dtfp',
                    t_int=t_int,
                    f_int=f_int,
                    eps=eps,
                    dtype=model_arr.dtype)

if __name__ == "__main__":

    zarr_store = zarr.DirectoryStore("zarr_data")
    zarr_group = zarr.hierarchy.group(store=zarr_store,
                                      overwrite=True,
                                      synchronizer=zarr.ThreadSynchronizer())
    zarr_compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=2)

    for nd in (1,):
        du = DaskArrays(nd=nd,nt=512,ct=32,nf=128,cf=32,na=64,ca=64)

        group_str = 'DIRECTION_%d' % nd
        dir_group = zarr_group.create_group(group_str)

        t_int, f_int, eps = 1.0, 1.0, 0.95

        jhr, jhjinv, flagcounts = compute_js(du, t_int, f_int, eps, FL.ILLCOND)
        update = compute_update(jhr, jhjinv)

        # Just sum over arrays so we don't run out of memory
        results = (jhr.sum(), jhjinv.sum(), update.sum(), flagcounts)

        gains = gain_update_loop(du.o, du.m, du.g, du.f, t_int, f_int, eps)

        # zarr chunks are slightly different and
        # not as flexible as dask chunks.
        # Set zarr chunk = max dask chunk for each dimension
        chunks = tuple(max(c) for c in gains.chunks)

        zarr_out = dir_group.empty("GAINS",
                                    shape=gains.shape,
                                    dtype=gains.dtype,
                                    chunks=chunks,
                                    compressor=zarr_compressor)

        # Final graph key is a zarr store operation
        store = da.store(gains, zarr_out, compute=False, lock=False)


        # Configure dask run options
        from multiprocessing.pool import ThreadPool
        from chest import Chest

        # 8 threads, spill to disk at memory limit
        optkw = {
            'pool' : ThreadPool(8),
            'cache' : Chest(available_memory=10e9)
        }

        # Task, CPU/Memory and Task Cache profilers
        progress = ProgressBar()
        prof = Profiler()
        rprof = ResourceProfiler()
        cprof = CacheProfiler()

        # Run the graph
        with progress, prof, rprof, cprof, da.set_options(**optkw):
            dask.compute(store)

        visualize([prof, rprof, cprof])

