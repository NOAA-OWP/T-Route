from setuptools import setup, find_namespace_packages
from distutils.extension import Extension
import sys
import numpy as np
from distutils.command.build_ext import build_ext
import os
import subprocess

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

# adapted from https://stackoverflow.com/a/5192738/489116 and https://stackoverflow.com/a/32192172/489116
# also of interest: https://stackoverflow.com/a/60954137/489116
fcompopt = {
    'intel': [],
    'gnu95' : ['-g']
}
flinkopt = {
    'intel': [],
    'gnu95' : []
}
flibs = {
    'intel': ['mpifort','mpi','ifcoremt','ifport','imf','svml','intlc'],
    'gnu95' : ['gfortran']
}

#new_fcompiler = fcompiler.new_compiler()
#fcompiler_type = fcompiler.compiler_type
# Open to better suggestions...
#fc = numpy.distutils.fcompiler.FCompiler()
#fc.customize()
#fc = fc.executables["compiler_f90"][0]
fc = os.environ['FC'] if 'FC' in os.environ else os.environ['F90'] if 'F90' in os.environ else subprocess.run(['which', 'fc'], capture_output=True).stdout.decode('UTF-8')[:-1]
result = subprocess.run([fc, '--version'], stdout=subprocess.PIPE)
result = result.stdout.decode('utf-8')
if "GNU" in result:
    fcompiler_type = 'gnu95'
elif "Intel" in result:
    fcompiler_type = 'intel'
else:
    raise Exception("Could not identify fortran compiler!")
print("Fortran compiler type is: {0}".format(fcompiler_type))

class build_ext_subclass( build_ext ):
    def build_extensions(self):
        for e in self.extensions:
            if fcompiler_type in fcompopt:
                e.extra_compile_args.extend(fcompopt[fcompiler_type])
            if fcompiler_type in flinkopt:
                e.extra_link_args.extend(flinkopt[fcompiler_type])
            if fcompiler_type in flibs:
                e.libraries.extend(flibs[fcompiler_type])
        build_ext.build_extensions(self)

reach = Extension(
    "troute.network.reach",
    sources=[
            "troute/network/reach.{}".format(ext),
            ],
    include_dirs=[np.get_include()],
    extra_objects=[],
    libraries=[],
    extra_compile_args=["-g"],
)

musk = Extension(
    "troute.network.musking.mc_reach",
    sources=[
            "troute/network/musking/mc_reach.{}".format(ext),
            ],
    include_dirs=[np.get_include(), "troute/network/"],
    extra_objects=[],
    libraries=[],
    extra_compile_args=["-g"],
)

levelpool_reservoirs = Extension(
    "troute.network.reservoirs.levelpool.levelpool",
    sources=[
             "troute/network/reservoirs/levelpool/levelpool.{}".format(ext),
             "troute/network/logger.c",
             ],
    include_dirs=[np.get_include(), "troute/network/"],
    extra_objects=["./libs/binding_lp.a"],
    libraries=["netcdff", "netcdf"],
    extra_compile_args=["-g"],
)

#hybrid_reservoirs = Extension(
#    "troute.network.reservoirs.hybrid.hybrid",
#    sources=[
#             "troute/network/reservoirs/hybrid/hybrid.{}".format(ext),
#             "troute/network/logger.c",
#             ],
#    include_dirs=[np.get_include(), "troute/network/"],
#    extra_objects=["./libs/bind_hybrid.a"],
#    libraries=["netcdff", "netcdf"],
#    extra_compile_args=["-g"],
#)

rfc_reservoirs = Extension(
    "troute.network.reservoirs.rfc.rfc",
    sources=[
             "troute/network/reservoirs/rfc/rfc.{}".format(ext),
             "troute/network/logger.c",
             ],
    include_dirs=[np.get_include(),  "troute/network/"],
    extra_objects=["./libs/bind_rfc.a"],
    libraries=["netcdff", "netcdf"],
    extra_compile_args=["-g"],
)

package_data = {"troute": ["__init__.pxd"],
                "troute.network": ["reach.pxd", "__init__.pxd", "reach_structs.h", "reach_structs.c", "logger.h"],
                "troute.network.musking": ["mc_reach.pxd", "__init__.pxd", "mc_reach_structs.h", "mc_reach_structs.c"],
                "troute.network.reservoirs":["__init__.pxd"],
                "troute.network.reservoirs.levelpool":["__init__.pxd", "levelpool.pxd", "levelpool_structs.h", "levelpool_structs.c"],
                "troute.network.reservoirs.hybrid":["__init__.pxd", "hybrid.pxd", "hybrid_structs.h", "hybrid_structs.c"],
                "troute.network.reservoirs.rfc":["__init__.pxd", "rfc.pxd", "rfc_structs.h", "rfc_structs.c"],
                 }
ext_modules = [reach, levelpool_reservoirs, rfc_reservoirs, musk]

if USE_CYTHON:
    from Cython.Build import cythonize

    ext_modules = cythonize(ext_modules, compiler_directives={"language_level": 3})

setup(
    name="troute.network",
    namespace_packages=["troute"],
    packages=['troute']+find_namespace_packages(include=["troute.*"]),#["troute.network", "troute.network.reservoirs", "troute.network.reservoirs.levelpool"],
    ext_modules=ext_modules,
    ext_package="",
    package_data=package_data,
    cmdclass = {'build_ext': build_ext_subclass }
)
