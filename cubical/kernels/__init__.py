#   Copyright 2020 Jonathan Simon Kenyon
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
import numpy as np
import traceback
import os.path
import importlib

# number of OMP threads to run, for OMP-related kernels
num_omp_threads = 0

use_parallel = False
use_cache = False

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

# _omp_import = [False]

def import_kernel(name):
    """ Imports named kernel and returns it. """

    return importlib.import_module("cubical.kernels." + name)





