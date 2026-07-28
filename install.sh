#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "--- Starting t-route Installation ---"

# If on a cluster platform that requires module load
# commands to load up python, cmake, netcdf, and gcc 
# or intel compilers and then go ahead and insert 
# those module commands here. Below are example 
# module load commands for the NOAA RDHPCS Ursa cluster
module purge

####### gcc/gfortran compilers ##############
#module load hpcx-mpi/2.18.1
#module load netcdf-fortran/4.6.1
#module load cmake/3.30.2
#module load python/3.11
#export FC=gfortran
#export CC=gcc
#export CXX=g++
############################################

####### intel compilers ####################
module load intel-oneapi-compilers/2025.1.1
module load intel-oneapi-mpi/2021.15.0
module load netcdf-fortran/4.6.1
module load cmake/3.30.2
module load python/3.11
export FC=mpiifx
export CC=mpiicx
export CXX=mpiicpx
############################################



# User defined installation pathway to the virtual Python
# for t-route needs to be set here! Otherwise, default is 
# in the t-route root directory
INSTALL_DIR=$(pwd)

# System dependency check to ensure we have all the required
# compilers and libraries for t-route
dependencies=("gcc" "gfortran" "cmake" "python3" "nc-config" "nf-config")

for tool in "${dependencies[@]}"; do
    if ! command -v "$tool" &> /dev/null; then
        echo "Error: $tool is not installed or not in PATH."
        echo "Please install the missing dependency before running the t-route installation script."
        exit 1
    fi
done

# Set compiler definitions based on user defined environmental variables 
# or defer to standard defaults
export FC=${FC:-gfortran}
export CC=${CC:-gcc}
export CXX=${CXX:-g++}


# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual Python environment for t-route..."
    python3 -m venv venv
fi


# Automatically extract paths from the system's NetCDF installation
NC_LIB_PATH=$(nc-config --prefix)/lib
NF_LIB_PATH=$(nf-config --prefix)/lib
NC_INC=$(nc-config --includedir)

# Set Linker flags so the compiler knows where the NetCDF binaries live
export LDFLAGS="-L${NC_LIB_PATH} -L${NF_LIB_PATH} -Wl,-rpath,${NC_LIB_PATH} -Wl,-rpath,${NF_LIB_PATH}"
export NETCDF="${NC_INC}"


echo "Installing t-route libraries and it's internal dependencies..."

# Activate Python virtual environment, install baseline
# Python dependencies for t-route libraries, and then 
# install the t-route libraries for Python
source $INSTALL_DIR/venv/bin/activate && \
$INSTALL_DIR/venv/bin/pip install --upgrade pip~=24.0 && \
$INSTALL_DIR/venv/bin/pip install -r requirements.txt && \
env LDFLAGS="$(nc-config --libs) $(nf-config --flibs)" NETCDF=$(nc-config --includedir) ./compiler.sh no-e

echo "--- Installation Complete! ---"

