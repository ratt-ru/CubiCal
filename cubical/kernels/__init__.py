# CubiCal: a radio interferometric calibration suite
# (c) 2017 Rhodes University & Jonathan S. Kenyon
# http://github.com/ratt-ru/CubiCal
# This code is distributed under the terms of GPLv2, see LICENSE.md for details
import numpy as np
import traceback
import os.path
import importlib

# number of OMP threads to run, for OMP-related kernels
num_omp_threads = 0

def allocate_reordered_array(shape, dtype, order, zeros=False):
    """
    Allocates an array with its dimensions laid out in a non-default order.
    
    Args:
        shape: shape of resulting array
        dtype: type
        order: order of dims in memory. E.g. [2,0,1] means the last dim changes slowest,
            then dim 0, then dim 1. If N=len(shape) < M=len(order), then the dims referred to
            by 'shape' are assumed to refer to the *trailing* N dimensions in 'order'.
            This allows 'order' to describe the layout of e.g. a (dir,model,time,freq,...) array,
            while the function can still be used to allocate a (time,freq,...) array.
        zeros: if True, array will be zeroed, else uninitialized. 

    Returns:
        View onto allocated array, of the given shape.
    """
    extra_dims = len(order) - len(shape)
    assert(extra_dims>=0)
    # take care of extra leading dimensions
    if extra_dims:
        slicer = tuple([0]*extra_dims)
        shape = [1]*extra_dims + list(shape)
    else:
        slicer = None

    _intrinsic_shape = [shape[i] for i in order]
    array = (np.zeros if zeros else np.empty)(_intrinsic_shape, dtype=dtype)
    array = array.transpose(np.argsort(order))

    if extra_dims:
        array = array[slicer]

    return array

_omp_import = [False]

def import_kernel(name):
    """
    Imports named kernel and returns it. If num_omp_threads>1, or this is a nested import from another OMP kernel, then
    tries to import the OMP version of the kernel, if available.
    
    In a normal cubical context, the only logic needed is dead simple: OMP is <=> num_omp_threads>1.
    All this other sophistication is to allow a program to import both an OMP and a non-OMP version of a kernel (and 
    therefore, of all the other kernels it references) simultaneously. This capability is needed by e.g. the 
    benchmarking code under test.
    """
    global _omp_import
    omp = False

    # if previous import was an OMP, or if threads>1, or name contains +_omp": force OMP version
    if _omp_import[-1] or num_omp_threads>1 or name.endswith("_omp"):
        omp = True
    # else check the callstack for an "_omp" moudle, this will also enable OMP
    else:
        for frame in traceback.extract_stack()[::-1]:
            caller = os.path.splitext(frame[0])[0]
            if "cubical/kernels" in caller and caller.endswith("_omp"):
                omp = True
                name += "_omp"
                break

    # try-finally is there to make sure the _omp_import stack is properly maintained during nested imports

    try:
        _omp_import.append(omp)
        # make list of kernels to try
        names = [name]
        if omp and not name.endswith("_omp"):
            names.insert(0, name+"_omp")
        # try to import kernel candidates. Return succssful one -- reraise exception if last one fails
        for i,name in enumerate(names):
            try:
                mod = importlib.import_module("cubical.kernels." + name)
                # print("{}import_kernel({})".format(" "*len(_omp_import), name))
                return mod
            except ImportError:
                if not name.endswith("_omp"):
                    raise
                print(("import_kernel({}): failed, trying fallback".format(name)))
    finally:
        _omp_import.pop()



