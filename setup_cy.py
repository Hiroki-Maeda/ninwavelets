from distutils.core import setup
from Cython.Build import cythonize
from numpy import get_include

setup(name='base',
      ext_modules=cythonize("nin_wavelets/base.pyx"),
      include_dirs=[get_include()]
      )
