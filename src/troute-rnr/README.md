# T-Route Replace and Route worker

This namespace package is meant to handle replace and route runs. Its purpose is to be run as a docker container within docker compose, or through IaC

to run through the main entrypoint, use:
```py
uv sync
uv run python main.py
```

to run through IaC use:
```py
uv sync --extra iac
uv run python main.py --iac
```
