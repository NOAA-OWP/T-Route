# Conecuh Test Case

The following file is meant to test the NHF on a test case in Alabama on USGS Gage 02374250. To run the test, do the following commands from the home directory:

```sh
uv venv --python 3.10
source .venv/bin/activate
./compiler.sh --uv
cd test/nhf/conecuh_case
uv pip install -r requirements.txt
uv run python -m nwm_routing -V5 -f test_case.yaml
```

# Streamflow Attribution

Streamflow estimates in the `channel_forcing/` folder were obtained from the following paper/dataset:

Song, Y., Bindas, T., Shen, C., Ji, H., Knoben, W. J. M., Lonzarich, L., et al. (2025). High-resolution national-scale water modeling is enhanced by multiscale differentiable physics-informed machine learning. Water Resources Research, 61, e2024WR038928. https://doi.org/10.1029/2024WR038928
