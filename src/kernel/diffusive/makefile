# compiler
FC := ${FC}

# compile flags
F90FLAGSGFORTRAN = -g -c -O2 -fPIC -fbounds-check 
F90FLAGSINTEL = -free -g -c -O2 -fPIC -I${NETCDF}/include 

FC_VERSION := $(shell $(FC) --version 2>&1 || true)

IS_INTEL := $(findstring Intel,$(FC_VERSION))$(findstring ifx,$(FC_VERSION))$(findstring ifort,$(FC_VERSION))

ifneq ($(strip $(IS_INTEL)),)
    FCFLAGS = $(F90FLAGSINTEL)
else
    FCFLAGS = $(F90FLAGSGFORTRAN)
endif

diffusive.o: diffusive.f90
	$(FC) $(FCFLAGS) -o $@ $<

pydiffusive.o: pydiffusive.f90
	$(FC) $(FCFLAGS) -o $@ $<

chxsec_lookuptable.o: chxsec_lookuptable.f90
	$(FC) $(FCFLAGS) -o $@ $<

pychxsec_lookuptable.o: pychxsec_lookuptable.f90
	$(FC) $(FCFLAGS) -o $@ $<


install:
	cp *.o ../../../src/troute-routing/troute/routing/fast_reach

clean:
	rm -f *.o *.mod
	rm -f ../../../src/troute-routing/troute/routing/fast_reach/diffusive.o
	rm -f ../../../src/troute-routing/troute/routing/fast_reach/pydiffusive.o
	rm -f ../../../src/troute-routing/troute/routing/fast_reach/diffusive.c
	rm -f ../../../src/troute-routing/troute/routing/fast_reach/diffusive.cpython*.so
	rm -f ../../../src/troute-routing/troute/routing/fast_reach/chxsec_lookuptable.o
	rm -f ../../../src/troute-routing/troute/routing/fast_reach/pychxsec_lookuptable.o
	rm -f ../../../src/troute-routing/troute/routing/fast_reach/chxsec_lookuptable.c
	rm -f ../../../src/troute-routing/troute/routing/fast_reach/chxsec_lookuptable.cpython*.so
