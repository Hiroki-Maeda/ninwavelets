from distutils.core import setup
from Cython.Build import cythonize
from numpy import get_include

setup(name='wavelets',
      ext_modules=cythonize("nin_wavelets/wavelets.pyx"),
      include_dirs=[get_include()]
      )
