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
import cubical
from setuptools import setup, find_packages

with open('README.md') as f:
    long_description = f.read()

try:
    import six
except ImportError:
    raise ImportError("Please install six before running install. If you're using pip 19 to install this package you should not be seeing this message.")

try:
    import numpy
except ImportError:
    raise ImportError("Please install numpy before running install. If you're using pip 19 to install this package you should not be seeing this message.")

# Check for readthedocs environment variable.

on_rtd = os.environ.get('READTHEDOCS') == 'True'

if on_rtd:
    requirements = ['numpy',
                    'futures; python_version <= "2.7"', 
                    'matplotlib',
                    'scipy']
else:
    requirements = ['future',
                    'numpy',
                    'llvmlite==v0.31.0; python_version <= "2.7"',
                    'numba==0.47.0; python_version <= "2.7"',
                    'numba; python_version >= "3.0"',
#                    'python-casacore<=3.0.0; python_version <= "2.7"',
                    'python-casacore',
                    'sharedarray >= 3.2.0',
                    'matplotlib<3.0',
                    'scipy',
                    'astro-tigger-lsm',
                    'six',
                    'futures; python_version <= "2.7"',
                    'astropy<3.0; python_version <= "2.7"',
                    'astropy>=3.0; python_version > "2.7"',
                    'psutil'
                    ]

setup(name='cubical',
      version=cubical.VERSION,
      description='Fast calibration implementation exploiting complex optimisation.',
      url='https://github.com/ratt-ru/CubiCal',
      lassifiers=[
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
      packages=find_packages(),
      python_requires='<3.0' if six.PY2 else ">=3.0", #build a py2 or py3 specific wheel depending on environment (due to cython backend)
      install_requires=requirements,
      include_package_data=True,
      zip_safe=False,
      scripts=['cubical/bin/print-cubical-stats',
               'cubical/bin/plot-leakage-solutions',
               'cubical/bin/plot-gain-solutions'],
      entry_points={'console_scripts': ['gocubical = cubical.main:main']},
      extras_require={
          'lsm-support': ['montblanc >= 0.6.1'],
          'degridder-support': ['ddfacet >= 0.5.0', 
                                'regions >= 0.4',
                                'meqtrees-cattery >= 1.7.0']
      }
)
