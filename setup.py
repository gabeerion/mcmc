from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

setup(
	name = "Ti/Tv code",
	ext_modules = cythonize('tt.pyx'),
	include_dirs = [np.get_include()]
)
