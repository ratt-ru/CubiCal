# CubiCal: a radio interferometric calibration suite
# (c) 2017 Rhodes University & Jonathan S. Kenyon
# http://github.com/ratt-ru/CubiCal
# This code is distributed under the terms of GPLv2, see LICENSE.md for details
import numpy as np

# number of OMP threads to run, for OMP-related kernels
num_omp_threads = 0

def allocate_reorded_array(shape, dtype, axis_memory_layout, zeros=False):
    """
    Allocates an array with its axes laid out in memory in a non-default order.
    axis_memory_layout gives the order of the axes in memory. E.g. [2,0,1] means the last axis changes slowest,
    then axis 0, then axis 1.
    """
    _intrinsic_shape = [shape[i] for i in axis_memory_layout]
    allocator = np.zeros if zeros else np.empty
    return allocator(_intrinsic_shape, dtype=dtype).transpose(np.argsort(axis_memory_layout))


