from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules = cythonize("level_function.pyx"),
    include_dirs=[numpy.get_include()]
)
