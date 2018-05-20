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
import glob

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

cmpl_args = ['-ffast-math',
             '-O2', 
             '-march=native',  
             '-mtune=native', 
             '-ftree-vectorize' ]

cmpl_args_omp = cmpl_args + ['-fopenmp']

link_args = []

link_args_omp = link_args + ['-lgomp']

if cythonize:

    log.info("Cython is available. Cythonizing...")

    CCO.buffer_max_dims = 9

    extensions = []
    for source in glob.glob("cubical/kernels/*.pyx"):
        name, ext = os.path.splitext(source)
        omp = name.endswith("_omp")
        # identify which kernels need to go via the C++ compiler
        cpp = any([x in name for x in "cytf_plane", "cyf_slope", "cyt_slope", "rebinning"])

        extensions.append(Extension(
            name.replace("/","."), [source],
            include_dirs=[include_path],
            extra_compile_args=cmpl_args_omp if omp else cmpl_args,
            extra_link_args=link_args_omp if omp else link_args,
            language="c++" if cpp else "c"
        ))

    extensions = cythonize(extensions, compiler_directives={'binding': True}, annotate=True)

else:

    log.info("Cython unavailable. Using bundled .c and .cpp files.")

    extensions = []
    for source in glob.glob("cubical/kernels/*.c") + glob.glob("cubical/kernels/*.cpp"):
        name, ext = os.path.splitext(source)
        omp = name.endswith("_omp")
        extensions.append(Extension(
            name.replace("/","."), [source],
            include_dirs=[include_path],
            extra_compile_args=cmpl_args_omp if omp else cmpl_args,
            extra_link_args=link_args_omp if omp else link_args
        ))


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
                    'cython',
                    'scipy',
                    'astro-tigger-lsm']

setup(name='cubical',
      version='1.0.0',
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
      packages=['cubical', 
                'cubical.machines', 
                'cubical.tools', 
                'cubical.kernels', 
                'cubical.plots',
                'cubical.database',],
      install_requires=requirements,
      include_package_data=True,
      zip_safe=False,
      ext_modules = extensions,
      entry_points={'console_scripts': ['gocubical = cubical.main:main']},           
)


