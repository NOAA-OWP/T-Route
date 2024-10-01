# compiler
FC := ${F90}

# compile flags
F90FLAGSGFORTRAN = -g -c -O2 -fPIC -fbounds-check 
F90FLAGSINTEL = -free -g -c -O2 -fPIC -I${NETCDF}/include 
ifeq ($(notdir ${FC}), ifort)
FCFLAGS = $(F90FLAGSINTEL)
else ifeq ($(notdir ${FC}), mpiifort)
FCFLAGS = $(F90FLAGSINTEL)
else
FCFLAGS = $(F90FLAGSGFORTRAN)
endif

diffusive.o: diffusive.f90 precis.mod
	$(FC) $(FCFLAGS) -o $@ $<

pydiffusive.o: pydiffusive.f90
	$(FC) $(FCFLAGS) -o $@ $<

chxsec_lookuptable.o: chxsec_lookuptable.f90 precis.mod
	$(FC) $(FCFLAGS) -o $@ $<

pychxsec_lookuptable.o: pychxsec_lookuptable.f90
	$(FC) $(FCFLAGS) -o $@ $<

diffusive_lightweight.o: diffusive_lightweight.f90 precis.mod
	$(FC) $(FCFLAGS) -o $@ $<

pydiffusive_lightweight.o: pydiffusive_lightweight.f90
	$(FC) $(FCFLAGS) -o $@ $<

precis.mod: varPrecision.f90
	$(FC) $(FCFLAGS) $<

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
	rm -f ../../../src/troute-routing/troute/routing/fast_reach/diffusive_lightweight.o
	rm -f ../../../src/troute-routing/troute/routing/fast_reach/pydiffusive_lightweight.o
	rm -f ../../../src/troute-routing/troute/routing/fast_reach/diffusive_lightweight.c
	rm -f ../../../src/troute-routing/troute/routing/fast_reach/diffusive_lightweight.cpython*.so

