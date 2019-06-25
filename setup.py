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
import glob

import cubical

from setuptools import setup, find_packages
from setuptools.extension import Extension
from setuptools.command.build_ext import build_ext
from setuptools import Command

with open('README.md') as f:
    long_description = f.read()

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
    raise ImportError("Please install cython before running install. If you're using pip 19 to install this package you should not be seeing this message.")

try:
    import six
except ImportError:
    raise ImportError("Please install six before running install. If you're using pip 19 to install this package you should not be seeing this message.")

try:
    import numpy
except ImportError:
    raise ImportError("Please install numpy before running install. If you're using pip 19 to install this package you should not be seeing this message.")

cmpl_args = ['-ffast-math',
             '-O2', 
             '-march=native',  
             '-mtune=native', 
             '-ftree-vectorize' ]

cmpl_args_omp = cmpl_args + ['-fopenmp']

link_args = []

link_args_omp = link_args + ['-lgomp']

# which extensions need to compile through C++ rather than C
cpp_extensions = "cytf_plane", "cyf_slope", "cyt_slope", "rebinning"


class gocythonize(Command):
    """ Cythonise CubiCal kernels. """
    
    description = 'Cythonise CubiCal kernels.'

    user_options = [('force', 'f', 'Force cythonisation.'),]

    def initialize_options(self):
        pass

    def finalize_options(self):
        self.force = self.force or 0

    @classmethod
    def populate_extensions(cls):
        if not cythonize:
            raise Exception("Cython not available, please install first.")
        
        CCO.buffer_max_dims = 9

        extensions = []
        
        for source in glob.glob("cubical/kernels/*.pyx"):
            name, ext = os.path.splitext(source)
            omp = name.endswith("_omp")
            # identify which kernels need to go via the C++ compiler
            cpp = any([x in name for x in cpp_extensions])

            extensions.append(
                Extension(name.replace("/","."), [source],
                          include_dirs=[include_path],
                          extra_compile_args=cmpl_args_omp if omp else cmpl_args,
                          extra_link_args=link_args_omp if omp else link_args,
                          language="c++" if cpp else "c"))
        
        return extensions

    def run(self):
        extensions = gocythonize.populate_extensions()
        cythonize(extensions, compiler_directives={'binding': True, 'language_level' : "3" if six.PY3 else "2"}, annotate=True, force=self.force)

# The default build_ext only builds extensions specified through the ext_modules list.
# However, to be absolutely safe for wheel building from source using pip v19, cythonization
# must be invoked to check that all the necessary cxx and c files have been created.
# If not, they need to be created before the extension modules are compiled using
# the normal cxx and c compiler invoked by the Extension class of setuptools.

class custom_build_ext(build_ext, gocythonize):
    """ Build all extension modules """
    
    description = 'Cythonise CubiCal kernels and build thereafter with the c/cxx compiler'

    user_options = [('force', 'f', 'Force cythonisation.')]

    def __init__(self, *args, **kwargs):
        build_ext.__init__(self, *args, **kwargs)
    
    def initialize_options(self):
        build_ext.initialize_options(self)
        gocythonize.initialize_options(self)

    def finalize_options(self):
        build_ext.finalize_options(self)
        gocythonize.finalize_options(self)
        
    def run(self):
        gocythonize.run(self) # first cythonize (if needed)
        build_ext.run(self) # then GNU build


c_cpp_extensions = []
for source in glob.glob("cubical/kernels/*.pyx"): 
    name, _ = os.path.splitext(source)
    is_cpp = any([s in name for s in cpp_extensions])
    is_omp = name.endswith("_omp")

    c_cpp_extensions.append(
        Extension(name.replace("/","."), [name + ".cpp" if is_cpp else name + ".c"],
                  include_dirs=[include_path],
                  extra_compile_args=cmpl_args_omp if is_omp else cmpl_args,
                  extra_link_args=link_args_omp if is_omp else link_args))

# Check for readthedocs environment variable.

on_rtd = os.environ.get('READTHEDOCS') == 'True'

if on_rtd:
    requirements = ['numpy',
                    'numba', 
                    'cython', 
                    'futures; python_version <= "2.7"', 
                    'matplotlib',
                    'scipy']
else:
    requirements = ['future',
                    'numpy',
                    'numba',
                    'python-casacore<=3.0.0; python_version <= "2.7"',
                    'python-casacore<=3.0.0; python_version >= "3.0"', 
                    'sharedarray @ git+https://gitlab.com/bennahugo/shared-array.git@master', 
                    'matplotlib<3.0',
                    'cython',
                    'scipy',
                    'astro-tigger-lsm',
                    'six',
                    'futures; python_version <= "2.7"'
                    ]

setup(name='cubical',
      version=cubical.VERSION,
      description='Fast calibration implementation exploiting complex optimisation.',
      url='https://github.com/ratt-ru/CubiCal',
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
      long_description=long_description,
      long_description_content_type='text/markdown',

      cmdclass={'build_ext': custom_build_ext,
                'gocythonize': gocythonize},

      packages=['cubical', 
                'cubical.data_handler',
                'cubical.machines',
                'cubical.tools', 
                'cubical.kernels', 
                'cubical.plots',
                'cubical.database',
                'cubical.madmax'],
      python_requires='<3.0' if six.PY2 else ">=3.0", #build a py2 or py3 specific wheel depending on environment (due to cython backend)
      install_requires=requirements,
      include_package_data=True,
      zip_safe=False,
      ext_modules = c_cpp_extensions,
      scripts = ['cubical/bin/print-cubical-stats'],
      entry_points={'console_scripts': ['gocubical = cubical.main:main']},
      extras_require={
          'lsm-support': ['montblanc @git+https://github.com/ska-sa/montblanc.git@0.6.1'],
      }
)


