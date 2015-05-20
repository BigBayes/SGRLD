from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

setup(
    cmdclass     = {'build_ext': build_ext},
    include_dirs = [np.get_include()], 
    ext_modules  = [
                    Extension("util_funcs", ["util_funcs.pyx"])
#                   , Extension("randcat", ["randcat.pyx"])
                    ]
)

