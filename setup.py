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
from setuptools import setup, find_packages
from setuptools.extension import Extension
from setuptools.command.install import install

USE_CYTHON = False
       
# try:
# 	import numpy as np
# 	include_path = np.get_include()
# except:
# 	print "Numpy failed to import."
# 	include_path = ""

def get_exts():

    import numpy as np
    include_path = np.get_include()

    if USE_CYTHON:

    	from Cython.Build import cythonize
    	import Cython.Compiler.Options as CCO	

    	CCO.buffer_max_dims = 9

    	extensions = \
    		[Extension(
    	        "cubical.kernels.cyfull_complex", ["cubical/kernels/cyfull_complex.pyx"],
    	        include_dirs=[include_path]),
    	     Extension(
    	        "cubical.kernels.cyphase_only", ["cubical/kernels/cyphase_only.pyx"],
    	        include_dirs=[include_path]),
    	     Extension(
    	        "cubical.kernels.cyfull_W_complex", ["cubical/kernels/cyfull_W_complex.pyx"],
    	        include_dirs=[include_path]),
    	     Extension(
    	        "cubical.kernels.cychain", ["cubical/kernels/cychain.pyx"],
    	        include_dirs=[include_path]),
    	     Extension(
    	        "cubical.kernels.cytf_plane", ["cubical/kernels/cytf_plane.pyx"],
    	        include_dirs=[include_path], language="c++"),
    	     Extension(
    	        "cubical.kernels.cyf_slope", ["cubical/kernels/cyf_slope.pyx"],
    	        include_dirs=[include_path], language="c++"),
    	     Extension(
    	        "cubical.kernels.cyt_slope", ["cubical/kernels/cyt_slope.pyx"],
    	        include_dirs=[include_path], language="c++")]

    	extensions = cythonize(extensions, compiler_directives={'binding': True})

        return extensions

    else:

    	extensions = \
    		[Extension("cubical.kernels.cyfull_complex", ["cubical/kernels/cyfull_complex.c"], 
    			include_dirs=[include_path]),
          	 Extension("cubical.kernels.cyphase_only", ["cubical/kernels/cyphase_only.c"], 
          	 	include_dirs=[include_path]),
          	 Extension("cubical.kernels.cyfull_W_complex", ["cubical/kernels/cyfull_W_complex.c"], 
          	 	include_dirs=[include_path]),
          	 Extension("cubical.kernels.cychain", ["cubical/kernels/cychain.c"], 
          	 	include_dirs=[include_path]),
          	 Extension("cubical.kernels.cytf_plane", ["cubical/kernels/cytf_plane.cpp"], 
          	 	include_dirs=[include_path]),
          	 Extension("cubical.kernels.cyf_slope", ["cubical/kernels/cyf_slope.cpp"], 
          	 	include_dirs=[include_path]),
          	 Extension("cubical.kernels.cyt_slope", ["cubical/kernels/cyt_slope.cpp"], 
          	 	include_dirs=[include_path])]

        return extensions

class custom_install(install):
    def run(self):
        os.system("python setup.py build_ext --inplace -f")
        install.run(self)

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
                    'python-casacore', 
                    'sharedarray', 
                    'matplotlib',
                    'scipy',
                    'astro-tigger']

print find_packages()

setup(name='cubical',
      version='0.2.5',
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
      cmdclass={'install': custom_install},  
      packages=['cubical', 'cubical.machines', 'cubical.tools', 'cubical.kernels'],
      setup_requires=['numpy'],
      install_requires=requirements,
      include_package_data=True,
      zip_safe=False,
      ext_modules = get_exts(),
      entry_points={'console_scripts': ['nocubical = cubical.main:main']},           
)
# scripts = ['cubical/bin/gocubical'],      
# scripts = ['cubical/bin/gocubical'],
#scripts=[os.path.join('cubical', script_name) for script_name in ['DDF.py', 'CleanSHM.py', 'MemMonitor.py', 'Restore.py', 'SelfCal.py']],
# cythonize(extensions, compiler_directives={'binding': True}),

