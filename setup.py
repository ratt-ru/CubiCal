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
from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
from setuptools.command.install import install
import numpy as np			

class custom_install(install):
	def run(self):
		os.system("python setup.py build_ext --inplace -f")
		install.run(self)

extensions = [Extension(
						"cubecal.kernels.cyfull_complex", ["cubecal/kernels/cyfull_complex.pyx"],
        				include_dirs=[np.get_include()], extra_compile_args=['-fopenmp', 
        				'-ffast-math', '-O2', '-march=native',	'-mtune=native', '-ftree-vectorize'],
						extra_link_args=['-lgomp']
					   ),
			  Extension(
						"cubecal.kernels.cyphase_only", ["cubecal/kernels/cyphase_only.pyx"],
        				include_dirs=[np.get_include()], extra_compile_args=['-fopenmp', 
        				'-ffast-math', '-O2', '-march=native',	'-mtune=native', '-ftree-vectorize'],
						extra_link_args=['-lgomp']
					   )]

setup(name='cubecal',
      version='0.2.1',
      description='Fast calibration implementation exploiting complex optimisation.',
      url='https://github.com/JSKenyon/phd-code',
      download_url='https://github.com/JSKenyon/phd-code/archive/0.2.1.tar.gz',
      classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: Ubuntu 14.04",
        "Programming Language :: Python",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Astronomy"],
      author='Jonathan Kenyon',
      author_email='jonosken@gmail.com',
      license='GNU GPL v3',
      cmdclass={'install': custom_install},	 
      packages=['cubecal', 'cubecal/tools', 'cubecal/kernels', 'cubecal/machines'],
      setup_requires=['numpy', 'cython'],
      install_requires=[  'numpy', 
                          'cython', 
                          'futures', 
                          'python-casacore', 
                          'sharedarray', 
                          'matplotlib'  ],
      include_package_data=True,
      package_data={'cubecal'	:	[	'main.py',
                									'solver.py', 
          		      							'statistics.py',
          		      							'plots.py',
          		      							'flagging.py',
          		      							'MBTiggerSim.py', 
          		      							'ReadModelHandler.py',
          		      							'TiggerSourceProvider.py', 
          		      							'bin/gocubecal'], 
                    'tools'		:	[	'cubecal/tools/logger.py', 
		                       			  'cubecal/tools/ModColor.py',
		                    		 	    'cubecal/tools/ClassPrint.py',
		                    		 	    'cubecal/tools/myoptparse.py',
		                    		 	    'cubecal/tools/parsets.py'],
                    'kernels'	:	[	'cubecal/kernels/cyfull_complex.pyx', 
                       					  'cubecal/kernels/cyphase_only.pyx'],
                    'machines'	:	[	'cubecal/machines/abstract_gain_machine.py', 
                       					    'cubecal/machines/complex_2x2_machine.py',
                       					    'cubecal/machines/phase_diag_machine.py'] },
      zip_safe=False,
      ext_modules = cythonize(extensions),
      scripts=['cubecal/bin/gocubecal'],
)

