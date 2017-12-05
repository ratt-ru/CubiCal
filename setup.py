# CubiCal: a radio interferometric calibration suite
# (c) 2017 Rhodes University & Jonathan S. Kenyon
# http://github.com/ratt-ru/CubiCal
# This code is distributed under the terms of GPLv2, see LICENSE.md for details
#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2017 SKA South Africa
#
# This file is part of CubeCal.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import os
import sys
import logging

from setuptools import setup, find_packages
from setuptools.extension import Extension
from setuptools.command.build_ext import build_ext

logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
log = logging.getLogger()

# Try get location of numpy headers. Compilation requires these headers. 

try:
    import numpy as np
except ImportError:
    include_path = ''
else:
    include_path = np.get_include()

# Use Cython if available.

try:
    from Cython.Build import cythonize
    import Cython.Compiler.Options as CCO
except ImportError:
    cythonize = None

cmpl_args = ['-fopenmp',
             '-ffast-math', 
             '-O2', 
             '-march=native',  
             '-mtune=native', 
             '-ftree-vectorize']
    
link_args = ['-lgomp']

if cythonize:

    log.info("Cython is available. Cythonizing...")

    CCO.buffer_max_dims = 9

    extensions = \
        [Extension(
            "cubical.kernels.cyfull_complex", ["cubical/kernels/cyfull_complex.pyx"],
            include_dirs=[include_path], extra_compile_args=cmpl_args, extra_link_args=link_args),
         Extension(
            "cubical.kernels.cyphase_only", ["cubical/kernels/cyphase_only.pyx"],
            include_dirs=[include_path], extra_compile_args=cmpl_args, extra_link_args=link_args),
         Extension(
            "cubical.kernels.cyfull_W_complex", ["cubical/kernels/cyfull_W_complex.pyx"],
            include_dirs=[include_path], extra_compile_args=cmpl_args, extra_link_args=link_args),
         Extension(
            "cubical.kernels.cychain", ["cubical/kernels/cychain.pyx"],
            include_dirs=[include_path], extra_compile_args=cmpl_args, extra_link_args=link_args),
         Extension(
            "cubical.kernels.cytf_plane", ["cubical/kernels/cytf_plane.pyx"],
            include_dirs=[include_path], language="c++", extra_compile_args=cmpl_args, 
            extra_link_args=link_args),
         Extension(
            "cubical.kernels.cyf_slope", ["cubical/kernels/cyf_slope.pyx"],
            include_dirs=[include_path], language="c++", extra_compile_args=cmpl_args, 
            extra_link_args=link_args),
         Extension(
            "cubical.kernels.cyt_slope", ["cubical/kernels/cyt_slope.pyx"],
            include_dirs=[include_path], language="c++", extra_compile_args=cmpl_args, 
            extra_link_args=link_args)]

    extensions = cythonize(extensions, compiler_directives={'binding': True})

else:

    log.info("Cython unavailable. Using bundled .c and .cpp files.")

    extensions = \
        [Extension("cubical.kernels.cyfull_complex", ["cubical/kernels/cyfull_complex.c"], 
            include_dirs=[include_path], extra_compile_args=cmpl_args, extra_link_args=link_args),
         Extension("cubical.kernels.cyphase_only", ["cubical/kernels/cyphase_only.c"], 
            include_dirs=[include_path], extra_compile_args=cmpl_args, extra_link_args=link_args),
         Extension("cubical.kernels.cyfull_W_complex", ["cubical/kernels/cyfull_W_complex.c"], 
            include_dirs=[include_path], extra_compile_args=cmpl_args, extra_link_args=link_args),
         Extension("cubical.kernels.cychain", ["cubical/kernels/cychain.c"], 
            include_dirs=[include_path], extra_compile_args=cmpl_args, extra_link_args=link_args),
         Extension("cubical.kernels.cytf_plane", ["cubical/kernels/cytf_plane.cpp"], 
            include_dirs=[include_path], extra_compile_args=cmpl_args, extra_link_args=link_args),
         Extension("cubical.kernels.cyf_slope", ["cubical/kernels/cyf_slope.cpp"], 
            include_dirs=[include_path], extra_compile_args=cmpl_args, extra_link_args=link_args),
         Extension("cubical.kernels.cyt_slope", ["cubical/kernels/cyt_slope.cpp"], 
            include_dirs=[include_path], extra_compile_args=cmpl_args, extra_link_args=link_args)]

# Check for readthedocs environment variable.

on_rtd = os.environ.get('READTHEDOCS') == 'True'

if on_rtd:
    requirements = ['numpy', 
                    'cython', 
                    'futures', 
                    'matplotlib',
                    'scipy']
else:
    requirements = ['numpy', 
                    'futures', 
                    'python-casacore>=2.1.2', 
                    'sharedarray', 
                    'matplotlib',
                    'scipy',
                    'astro-tigger']

setup(name='cubical',
      version='0.9.2',
      description='Fast calibration implementation exploiting complex optimisation.',
      url='https://github.com/JSKenyon/phd-code',
      classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering :: Astronomy"],
      author='Jonathan Kenyon',
      author_email='jonosken@gmail.com',
      license='GNU GPL v3',
      cmdclass={'build_ext': build_ext},
      packages=['cubical', 'cubical.machines', 'cubical.tools', 'cubical.kernels', 'cubical.plots'],
      install_requires=requirements,
      include_package_data=True,
      zip_safe=False,
      ext_modules = extensions,
      entry_points={'console_scripts': ['gocubical = cubical.main:main']},           
)


