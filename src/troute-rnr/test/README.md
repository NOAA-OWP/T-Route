# T-route Tests

Follow t-route readme to download dependencies and compile

```
Make venv in base repo
uv pip install -e .[test]
./compiler.sh
cd src/troute-rnr
uv pip install .
Download all data from s3:/hydrofabric-data/icefabric
At repo root, create data/parquet and data/warehouse.
Put downloaded icefabric data ino data/parquet
cd to troute-rnr , pytest test
```
