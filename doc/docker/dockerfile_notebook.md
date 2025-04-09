# Dockerfile.Notebook

This document describes the Docker setup for running JupyterLab with mounted volumes for development and analysis.

## Container Overview

The container provides a JupyterLab environment with:
- Python environment for data analysis
- Web interface accessible via port 8000

This container is a great way to run examples and integrated tests

## Docker Configuration

### Dockerfile
The Dockerfile sets up:
- Base Python environment
- JupyterLab installation
- Volume mount points for data and code
- Port 8000 exposed for web interface
- Working directory configuration

### Getting Started

Build:
```bash
docker build -t troute-notebook -f docker/Dockerfile.notebook .
```

Run:
```bash
docker run -p 8000:8000 troute-notebook
```

Then, take the URL from the output and put that into your browser. An example one is below:
```
http://127.0.0.1:8000/lab?token=<token>
```
