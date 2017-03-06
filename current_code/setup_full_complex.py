from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

setup(
	cmdclass = {'build_ext': build_ext},
    	ext_modules=[Extension("cyfull", ["cyfull_complex.pyx"],
               	include_dirs=[np.get_include()],
		extra_compile_args=['-fopenmp','-ffast-math','-o2','-march=native',
							'-mtune=native', '-ftree-vectorize'],
		extra_link_args=['-lgomp'])]
	)
