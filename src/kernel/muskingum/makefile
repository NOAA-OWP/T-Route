# compiler
FC := $(F90)

# Architecture tuning. ``-march=native`` / ``-mcpu=native`` makes the build
# host-specific: the resulting binary will SIGILL on any CPU that lacks an
# instruction the build host emitted. That is fine for a local/profiling
# build, but unsafe for portable artifacts (conda or pixi packages built on
# one machine and installed on another, container images run on a different
# CPU generation than the build host, HPC login-node builds executed on a
# different compute-node CPU).
#
# Default: portable build (``-O3 -funroll-loops`` only).
# Opt-in: ``make TROUTE_NATIVE=1`` to add the host-specific arch flag (used
# by benchmark images via the benchmark Dockerfile / compiler invocation).
# Override entirely with ``make ARCHFLAGS='-march=...'`` if you need a
# specific generation (e.g. ``-march=znver3`` for WCOSS2 EPYC).
ARCH := $(shell uname -m)
ifeq ($(TROUTE_NATIVE),1)
ifeq ($(ARCH),arm64)
ARCHFLAGS ?= -mcpu=native
else ifeq ($(ARCH),aarch64)
ARCHFLAGS ?= -mcpu=native
else
ARCHFLAGS ?= -march=native
endif
else
ARCHFLAGS ?=
endif

# compile flags
#FCFLAGS = -c -fdefault-real-8 -fno-align-commons -fbounds-check --free-form
F90FLAGSGFORTRAN = -g -c -O3 -funroll-loops $(ARCHFLAGS) -fPIC -lgfortran -lgcc -static-libgfortran -static-libgcc -nodefaultlibs
F90FLAGSINTEL = -free -w -c -O2 -fPIC -fconvert=big-endian -frecord-marker=4
ifeq ($(notdir ${FC}), ifort)
F90FLAGS = $(F90FLAGSINTEL)
else ifeq ($(notdir ${FC}), mpiifort)
F90FLAGS = $(F90FLAGSINTEL)
else
F90FLAGS = $(F90FLAGSGFORTRAN)
endif


.PHONY: reach all install
all: reach 

reach: mc_single_seg.o pymc_single_seg.o

mc_single_seg.o: MCsingleSegStime_f2py_NOLOOP.f90 precis.mod
	$(FC) $(F90FLAGS) -o $@ $<

pymc_single_seg.o: pyMCsingleSegStime_NoLoop.f90
	$(FC) $(F90FLAGS) -o $@ $<

precis.mod: varPrecision.f90
	$(FC) $(F90FLAGS) $<

install:
	cp *.o ../../../src/troute-routing/troute/routing/fast_reach

%.o: %.F
	$(FC) $(F90FLAGS) -o $@ $<

%.o: %.f90
	$(FC) $(F90FLAGS) -o $@ $<

clean:
	rm -f *.o *.mod
