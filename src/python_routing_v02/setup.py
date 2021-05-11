from setuptools import setup
from distutils.extension import Extension
import sys
import numpy as np

"""
If source ships with the cython generated .c code, then cython isn't a hard requirement
This setup can use if if passed the --use-cython flag, otherwise it looks for the c source
and builds using distutils.
"""

if "--use-cython" in sys.argv:
    USE_CYTHON = True
    sys.argv.remove("--use-cython")
else:
    USE_CYTHON = False

ext = "pyx" if USE_CYTHON else "c"

extra_compile_args = ["-g"]

reach = Extension(
    "reach",
    sources=["fast_reach/reach.{}".format(ext)],
    extra_objects=["fast_reach/mc_single_seg.o", "fast_reach/pymc_single_seg.o"],
    extra_compile_args=extra_compile_args,
)

mc_reach = Extension(
    "fast_reach",
    sources=["fast_reach/mc_reach.{}".format(ext), "fast_reach/muskingumcunge.c"],
    include_dirs=[np.get_include()],
    libraries=[],
    library_dirs=[],
    extra_objects=[],
    extra_compile_args=extra_compile_args,
)

diffusive = Extension(
    "diffusive",
    sources=["fast_reach/diffusive.{}".format(ext)],
    extra_objects=["fast_reach/diffusive.o", "fast_reach/pydiffusive.o"],
    include_dirs=[np.get_include()],
    extra_compile_args=extra_compile_args,
    libraries=["gfortran"],
)

ext_modules = [reach, mc_reach, diffusive]

if USE_CYTHON:
    from Cython.Build import cythonize

    ext_modules = cythonize(ext_modules, compiler_directives={"language_level": 3})

setup(
    name="compute_network_mc", ext_modules=ext_modules,
)
