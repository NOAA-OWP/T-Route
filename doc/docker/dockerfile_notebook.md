# JupyterLab via the dev image

This document describes running JupyterLab against t-route for development and analysis.

The separate `docker/Dockerfile.notebook` is gone. Its job is now the `dev` target of the
multi-target `docker/Dockerfile.dev`, which installs t-route with the `test` and `jupyter`
extras on top of the same base the production image uses, so the notebook environment and
the routing environment cannot drift apart.

## Container Overview

The `dev` target provides:
- t-route installed editable, with the compiled MC kernel
- the Python analysis stack and JupyterLab
- port 8000 exposed for the web interface

This is a good way to run the examples and the integration notebooks.

## Getting Started

Build:
```bash
docker build -t troute-dev -f docker/Dockerfile.dev --target dev .
```

Run. The image's default entrypoint is `nwm_routing`, so override it to start JupyterLab:
```bash
docker run --rm -p 8000:8000 --entrypoint /opt/venv/bin/jupyter troute-dev \
  lab --ip 0.0.0.0 --port 8000 --no-browser --allow-root
```

Then take the URL from the output and open it in your browser, for example:
```
http://127.0.0.1:8000/lab?token=<token>
```

## Related targets

`docker/Dockerfile.dev` also builds:

- `--target troute`, the production image, entrypoint `python -m nwm_routing`. This is what
  the CI image build publishes.
- `--target dev` with `--build-arg TROUTE_NATIVE=1` for a host-tuned benchmark build. The
  default of `0` keeps the kernel portable across machines.
