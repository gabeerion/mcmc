from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
	Extension('tvti', ['tt.pyx'],
		include_dirs=[np.get_include()]),
	Extension('impute', ['impute.pyx'],
		include_dirs=[np.get_include()]),
	]

setup(
	name = "MCMC code",
	ext_modules = cythonize(extensions),
)
