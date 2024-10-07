NETCDFINC := -I$(NETCDFINC) $(shell nf-config --fflags)
NETCDFLIB := $(shell nf-config --flibs)
COMPILER90 = $(F90)

F90FLAGSGFORTRAN = -w -c -ffree-form -ffree-line-length-none -fconvert=big-endian -frecord-marker=4 -static-libgfortran -lgfortran -lgcc -static-libgcc 
F90FLAGSINTEL = -free -w -c -O2 -fconvert=big-endian -frecord-marker=4
ifeq ($(notdir ${FC}), ifort)
F90FLAGS = $(F90FLAGSINTEL)
else ifeq ($(notdir ${FC}), mpiifort)
F90FLAGS = $(F90FLAGSINTEL)
else
F90FLAGS = $(F90FLAGSGFORTRAN)
endif

F90FLAGS := -g $(F90FLAGS) $(NETCDFINC) $(NETCDFLIB)  -nodefaultlibs -fPIC
ARFLAGS = rc

VPATH = ./Level_Pool ./Persistence_Level_Pool_Hybrid ./RFC_Forecasts

.PHONY: all install install_lp

all: binding_lp.a bind_hybrid.a bind_rfc.a

test_main: test_main.o binding_lp.a bind_hybrid.a bind_rfc.a
	$(COMPILER90) -g -o $@ $^

test_main.o: test_binding.c
	$(COMPILER90) -g -c -o $@ $<

module_levelpool.o: module_reservoir.o module_levelpool_properties.o module_levelpool_state.o

#module_persistence_levelpool_hybrid.o: module_reservoir.o module_hydro_stop.o module_reservoir_utilities.o # # # # # #module_persistence_levelpool_hybrid_properties.o module_persistence_levelpool_hybrid_state.o module_levelpool.o #module_reservoir_read_timeslice_data.o

module_rfc_forecasts.o: module_reservoir.o module_hydro_stop.o module_reservoir_utilities.o module_rfc_forecasts_properties.o module_rfc_forecasts_state.o module_levelpool.o module_reservoir_read_rfc_time_series_data.o

bind_lp.o: module_levelpool.o

#bind_hybrid.o: module_persistence_levelpool_hybrid.o

bind_rfc.o: module_rfc_forecasts.o

binding_lp.a: module_reservoir.o module_levelpool_properties.o module_levelpool_state.o module_levelpool.o bind_lp.o
	$(AR) $(ARFLAGS) $@ $^

#bind_hybrid.a: module_reservoir.o module_hydro_stop.o module_reservoir_utilities.o module_persistence_levelpool_hybrid_properties.o #module_persistence_levelpool_hybrid_state.o module_levelpool_properties.o module_levelpool_state.o module_levelpool.o #module_reservoir_read_timeslice_data.o module_persistence_levelpool_hybrid.o bind_hybrid.o
#	$(AR) $(ARFLAGS) $@ $^

bind_rfc.a: module_reservoir.o module_hydro_stop.o module_reservoir_utilities.o module_rfc_forecasts_properties.o module_rfc_forecasts_state.o module_levelpool_properties.o module_levelpool_state.o module_levelpool.o module_reservoir_read_rfc_time_series_data.o module_rfc_forecasts.o bind_rfc.o
	$(AR) $(ARFLAGS) $@ $^

%.o: %.F
	$(COMPILER90) $(F90FLAGS) $(NETCDFLIB) -c -o $@ $<

%.o: %.f90
	$(COMPILER90) $(F90FLAGS) $(NETCDFLIB) -c -o $@ $<

#install: binding_lp.a bind_hybrid.a bind_rfc.a
#install: binding_lp.a bind_rfc.a
#	echo $^
#	for lib in $^; do \
#		cp $$lib ../../troute-network/libs; \
#	done

install_lp: binding_lp.a
	cp $< ../../troute-network/libs;
    
install_rfc: bind_rfc.a
	cp $< ../../troute-network/libs;

clean:
	rm -f *.o
	rm -f *.mod
	rm -f *.a
	rm -f reservoir_tests
	rm -f test_main
