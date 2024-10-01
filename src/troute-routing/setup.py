from setuptools import setup, find_namespace_packages
from distutils.extension import Extension
import sys
import numpy as np
from distutils.command.build_ext import build_ext
import os
import subprocess

import troute.network as tn
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
    "troute.routing.fast_reach.reach",
    sources=["troute/routing/fast_reach/reach.{}".format(ext)],
    extra_objects=[
        "troute/routing/fast_reach/mc_single_seg.o",
        "troute/routing/fast_reach/pymc_single_seg.o",
    ],
    extra_compile_args=["-O2", "-g"],
    libraries=[],
)
_top_path = tn.__path__[0]
_subs = [_top_path+sub for sub in ['/musking', '/reservoirs'] ]
_sub_subs = [_top_path+'/reservoirs/'+sub for sub in ['levelpool', 'rfc', 'hybrid'] ]
_include_paths = [ np.get_include(), _top_path ] + _subs + _sub_subs
mc_reach = Extension(
    "troute.routing.fast_reach.mc_reach",
    sources=["troute/routing/fast_reach/mc_reach.{}".format(ext)],
    include_dirs=_include_paths,
    libraries=[],
    library_dirs=[],
    extra_objects=[],
    extra_compile_args=["-O2", "-g"],
)

simple_da = Extension(
    "troute.routing.fast_reach.simple_da",
    sources=[
        "troute/routing/fast_reach/simple_da.{}".format(ext),
    ],
    include_dirs=[np.get_include()],
    libraries=[],
    library_dirs=[],
    extra_objects=[],
    extra_compile_args=["-O2", "-g"],
)

diffusive = Extension(
    "troute.routing.fast_reach.diffusive",
    sources=["troute/routing/fast_reach/diffusive.{}".format(ext)],
    extra_objects=[
        "troute/routing/fast_reach/diffusive.o",
        "troute/routing/fast_reach/pydiffusive.o",
    ],
    include_dirs=[np.get_include()],
    extra_compile_args=["-O2", "-g"],
    libraries=[],
)

hybrid_routing_reach = Extension(
    "troute.routing.fast_reach.hybrid_routing_reach",
    sources=["troute/routing/fast_reach/hybrid_routing_reach.{}".format(ext)],
    include_dirs=_include_paths,
    libraries=[],
    library_dirs=[],
    extra_objects=[],
    extra_compile_args=["-O2", "-g"],
)

chxsec_lookuptable = Extension(
    "troute.routing.fast_reach.chxsec_lookuptable",
    sources=["troute/routing/fast_reach/chxsec_lookuptable.{}".format(ext)],
    extra_objects=[
        "troute/routing/fast_reach/chxsec_lookuptable.o",
        "troute/routing/fast_reach/pychxsec_lookuptable.o",
    ],
    include_dirs=[np.get_include()],
    extra_compile_args=["-O2", "-g"],
    libraries=[],
)

diffusive_lightweight = Extension(
    "troute.routing.fast_reach.diffusive_lightweight",
    sources=["troute/routing/fast_reach/diffusive_lightweight.{}".format(ext)],
    extra_objects=[
        "troute/routing/fast_reach/diffusive_lightweight.o",
        "troute/routing/fast_reach/pydiffusive_lightweight.o",
    ],
    include_dirs=[np.get_include()],
    extra_compile_args=["-O2", "-g"],
    libraries=[],
)

package_data = {"troute.fast_reach": ["reach.pxd", "fortran_wrappers.pxd", "utils.pxd"]}
ext_modules = [reach, mc_reach, diffusive, simple_da, hybrid_routing_reach, chxsec_lookuptable, diffusive_lightweight]

if USE_CYTHON:
    from Cython.Build import cythonize

    ext_modules = cythonize(ext_modules, compiler_directives={"language_level": 3})

setup(
    name="troute.routing",
    namespace_packages=["troute"],
    packages=find_namespace_packages(
        include=["troute.*"]
    ),  # ["troute.fast_reach", "troute.routing"],
    ext_modules=ext_modules,
    ext_packages="",
    package_data=package_data,
    py_modules=["build_tests"],
    cmdclass = {'build_ext': build_ext_subclass }
)
