# T-Route Replace and Route worker

This namespace package is meant to handle replace and route runs. Its purpose is to be run as a docker container within docker compose, or through IaC

# Dependencies:

### Icefabric
This repo depends on access to the Raytheon Icefabric package. To install, from github, please use the following installation
```sh
uv pip install git+https://github.com/NGWPC/icefabric.git#subdirectory=src/icefabric_tools
```

if this command does not work, you will need to access the pip wheels, which can be located in the following location: https://github.com/NGWPC/hydrovis/tree/pi_6/Source/RnR/dist
