# CubiCal: a radio interferometric calibration suite
# (c) 2017 Rhodes University & Jonathan S. Kenyon
# http://github.com/ratt-ru/CubiCal
# This code is distributed under the terms of GPLv2, see LICENSE.md for details

num_omp_threads = 0

def import_cyfull_complex():
    """Imports and return OMP or non-OMP kernel as appropriate"""
    if num_omp_threads > 1:
        import cyfull_complex_omp
        return cyfull_complex_omp
    else:
        import cyfull_complex
        return cyfull_complex

