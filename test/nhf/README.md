# NHF Tests

Tests for the NHF (NextGen Hydrofabric) routing implementation and the t-route V5 CLI.

- **Integration tests** — full model runs, marked `integration` for pytest. Each test checks whether its input data exists and skips if not. Run `prep_tests.py` first.
- **Unit tests** — partially broken, see below.
- **Diagnostic suite** — standalone scripts for plotting and metrics. Not collected by pytest.

BMI execution (`utils/run_bmi.py`) is currently broken.

---

## Quick Start

> [!WARNING]
> Running these commands will create some indices in the flowpath table and update the lakes table directly on your source geopackage.  If any of your workflows depend on hashes of the source file, be warned!

### 1 — Build test data

```console
# All tests (uses default NHF geopackage at /t-route/nhf_1.2.1.gpkg)
python -m test.nhf.prep_tests

# All tests with explicit NHF geopackage path
python -m test.nhf.prep_tests --nhf-gpkg /path/to/nhf.gpkg

# Specific tests only
python -m test.nhf.prep_tests --nhf-gpkg /path/to/nhf.gpkg --test conecuh patuxent

# Re-generate existing data
python -m test.nhf.prep_tests --nhf-gpkg /path/to/nhf.gpkg --refresh
```

### 2 — Run tests

```console
# All tests (unit + integration)
pytest test/nhf

# Integration tests only
pytest -m integration

# Unit tests only
pytest -m "not integration" test/nhf
```

### 3 — Plot test hydrographs (optional)

```console
# All test cases
python -m test.nhf.plot_tests

# Specific test cases
python -m test.nhf.plot_tests --test conecuh ciss_creek
```

---

## Integration Tests

Each test runs t-route end-to-end and checks model output.  If the input data is missing the test is skipped — run `prep_tests.py` first.

### Conecuh River (`conecuh_case`)

Base NHF test, routing on an Alabama basin (USGS gage 02374250) for a December 2009 flood.

Passing criteria: Peak flow at outlet is within acceptable range.

| Parameter | Value |
|---|---|
| Outlet fp_id | `1270581653591645` |
| Period | 2009-12-12 to 2009-12-29 |
| Forcing | `retro` |
| Lat / Lon | 31.06503,-87.06368 |

---

### Patuxent Reservoir (`patuxent`)

Simple, well-gauged site for checking level-pool reservoir behaviour.

Passing criteria: Peak outflow upstream and downstream of one of the reservoirs is within acceptable range.

| Parameter | Value |
|---|---|
| Outlet fp_id | `1284196257037837` |
| Period | 2011-09-05 to 2011-09-15 |
| Forcing | `retro` |
| Lat / Lon | 38.91812,-76.68369 |

---

### Ciss Creek (`ciss_creek`)

Single flowpath with four reservoirs including two on the main flowpath and two on a shared virtual flowpath.  Checks

 - level pool routing is ocurring on virtual flowpaths
 - many reservoirs will route when on the same flowpath or virtual flowpath
 - lakeout file is usable

Passing criteria: Peak outflow is within acceptable range.

| Parameter | Value |
|---|---|
| Outlet fp_id | `1288454913281725` |
| Period | 2000-01-01 to 2000-01-03 |
| Forcing | `pulse` |
| Lat / Lon | 46.26594,-69.58566 |

---

### Great Lakes (`great_lakes`)

DA-forced outflows from four fp_id-bearing Great Lakes (Superior, Huron-Michigan, Erie, Ontario). Uses USGS timeslice files, Canadian timeslice files, and a Lake Ontario outflow CSV. Checks that forced values propagate correctly downstream. A domain is committed to the repo and runs as-is; use `prep_tests.py --test great_lakes --refresh` with a newer NHF geopackage to regenerate.

Passing criteria: Flows at outlets of lakes match DA values very closely

| Parameter | Value |
|---|---|
| Lake fp_ids | `1278348162056612` (Superior), `1276364270499315` (Huron-Michigan), `1286192735893685` (Erie), `1287248237297035` (Ontario) |
| Period | 2000-01-01 to 2000-01-03 |
| Forcing | Zero qlat + USGS/Canada DA timeslice files + Lake Ontario outflow CSV |

---

### Four Lakes (`four_lakes`)

Runs all four reservoir DA types in one shot: USGS persistence (type 2), USACE persistence (type 3), RFC time-series (type 4), USBR persistence (type 7). `prep_tests.py` generates synthetic constant-flow DA files. The test checks that the immediately-downstream reach of each reservoir carries the expected outflow. This test also tests the lakeout functionality.

Passing criteria: Flows at outlets of lakes (which have been forced low) match DA values very closely

| Parameter | Value |
|---|---|
| Outlet fp_id | `1276182780176988` |
| Period | 2020-01-01 00:00 to 01:00 |
| Forcing | Constant qlat + synthetic DA files |
| Lat / Lon | 42.90326,-89.21309 |

---

## Unit Tests

| File | Coverage |
|---|---|
| `test_flow_scaling.py` | Catchment discharge distribution to NHF links |
| `test_flow_scaling_utils.py` | Flow scaling utility functions (`expand_flow_scaling_to_flowveldepth`, `merge_routing_and_nonrouting_results`, `create_nonrouting_run_result`) |
| `test_nhf_utils.py` | NHF network topology utilities (`build_downstream_connections`, `build_upstream_terminal`, `find_headwaters`, `find_tailwaters`, `validate_connections`) |

**`test_flow_scaling.py` is currently broken.** `distribute_catchment_discharge` was removed from `troute.nhf_discretize` and the tests were not updated. `test_flow_scaling.py` will fail at import until this is fixed.

---

## Diagnostic Suite

`utils/generate_diagnostics.py` computes a standard set of hydraulic metrics for any completed run. It exists because raw NetCDF output is hard to QC — routing bugs often produce reasonable-looking hydrographs while quietly violating volume conservation or generating bad Courant numbers.

```console
python test/nhf/utils/generate_diagnostics.py -f /path/to/run.yaml
python test/nhf/utils/generate_diagnostics.py -f /path/to/run.yaml --n-samples 500
```

See [Diagnostic Suite Specifics](#diagnostic-suite-specifics) for metric details.

---

# CLI Reference

## `prep_tests.py`

Builds test input data without running the tests. Skips existing outputs unless `--refresh` is passed.

```
python -m test.nhf.prep_tests [OPTIONS]

Options:
  --nhf-gpkg PATH        NHF geopackage path (default: /t-route/nhf_1.2.1.gpkg).
  --test NAME [NAME ...] Which tests to prep. Default: all.
                         Choices: conecuh patuxent ciss_creek great_lakes
                                  four_lakes
  --refresh              Regenerate even if outputs already exist.
```

## `plot_tests.py`

Generates hydrograph PNGs for every reach monitored by each test case's `PEAK_BOUNDS`.

```
python -m test.nhf.plot_tests [OPTIONS]

Options:
  --test NAME [NAME ...] Which tests to plot. Default: all.
                         Choices: conecuh patuxent ciss_creek great_lakes
                                  four_lakes
```

## `utils/subset_nhf.py`

Pulls all NHF components upstream of an outlet flowpath into a new geopackage. Called by `prep_tests.py` automatically.

```
python test/nhf/utils/subset_nhf.py [OPTIONS]

Options:
  --source-gpkg PATH     Source NHF geopackage.
  --out-gpkg PATH        Output geopackage path.
  --outlet-fp-id INT     fp_id of the outlet flowpath.
```

## `utils/make_forcing.py`

Writes per-timestep lateral inflow CSVs for a given case.

```
python test/nhf/utils/make_forcing.py [OPTIONS]

Options:
  --hf-path PATH         Path to the NHF GeoPackage.
  --forcing-dir PATH     Directory to write per-timestep forcing CSVs.
  --start-time STR       Simulation start (e.g. "2009-12-12 00:00").
  --end-time STR         Simulation end.
  --forcing-mode MODE    retro | pulse | constant  (default: retro)
  --peak-qlat FLOAT      Peak discharge m³/s for pulse mode (default: 10000).
  --constant-qlat FLOAT  Constant qlat m³/s for constant mode (default: 1.0).
  --runout-period INT    Hours of zero-qlat runout to append after end-time (default: 0).
```

| Mode | Description |
|---|---|
| `retro` | Hourly lateral inflows from the NWM v3.0 retrospective Zarr on S3. Requires a `reference_flowpaths` layer in the gpkg. |
| `pulse` | Synthetic unit-hydrograph pulse scaled to `--peak-qlat`, applied uniformly to all reaches. Shape is resampled to fit `[start-time, end-time]`. |
| `constant` | Flat `--constant-qlat` on every reach for every timestep. |

## `utils/make_configs.py`

Generates a t-route YAML configuration file from a `Config` dataclass. Primarily used internally by test setup functions.

```
python test/nhf/utils/make_configs.py [OPTIONS]

Options:
  --root-dir PATH        Root directory for the test case.
  --start-time STR       Simulation start time.
  --end-time STR         Simulation end time.
```

## `utils/make_da.py`

Generates synthetic DA (data assimilation) forcing files. Supports persistence DA types (USGS, USACE, USBR, Canada/WSC) as 15-minute timeslice NetCDFs, and RFC forecast files as hourly per-station NetCDFs.

```
python test/nhf/utils/make_da.py [OPTIONS]

Options:
  --da-type TYPE         DA type: usgs | usace | usbr | canada | rfc.
  --station-ids ID ...   One or more station identifiers.
  --start-time STR       Simulation start time.
  --end-time STR         Simulation end time.
  --output-dir PATH      Root output directory (default: reservoir_da/).
  --discharge FLOAT      Constant discharge value m³/s (default: 1.0).
  --discharge-quality INT  Quality flag (default: 100).
  --rfc-lookback-hours INT  Hours of observed lookback in RFC files (default: 28).
  --rfc-forecast-hours INT  Hours of synthetic forecast in RFC files (default: 12).
```

## `utils/generate_reference_data.py`

Fetches NWM v3.0 retrospective streamflow and USGS observed discharge for all active gages in a hydrofabric. Writes `gage_reference_data.nc`.

```
python test/nhf/utils/generate_reference_data.py [OPTIONS]

Options:
  --config PATH          Path to the t-route config YAML.
  --output-dir PATH      Directory to write gage_reference_data.nc.
  --dv-only              Skip instantaneous-value (IV) requests; derive all values
                         from daily means interpolated to the retrospective time index.
```

## `utils/generate_diagnostics.py`

Computes hydraulic QC metrics on a random sample of reaches from a completed run. See [Diagnostic Suite Specifics](#diagnostic-suite-specifics).

```
python test/nhf/utils/generate_diagnostics.py [OPTIONS]

Options:
  -f, --file PATH        Path to t-route config YAML.
  -n, --n-samples INT    Number of flowpaths to sample (default: 500).
```

## `utils/run_bmi.py`

Runs a t-route config file through the BMI interface. **Currently broken.**

```
python test/nhf/utils/run_bmi.py [OPTIONS]

Options:
  --config-file PATH     Path to the config YAML.
```

---

# Diagnostic Suite Specifics

`generate_diagnostics.py` runs on a random sample of reaches for efficiency. Sample size is set with `--n-samples`. Reaches are drawn as follows:

- 5% shortest reaches
- 5% longest reaches
- 5% steepest reaches
- 5% shallowest reaches
- remaining 80% distributed across stream orders

NB: example graphics below are from a 500-reach sample of the conecuh_case run at commit 4b3a98054daafa942fd94bf3de6d4783b207a4ca.

### Per-Reach Data Collection

- Upstream inflows ($Q_{us}$)
- Lateral inflows ($Q_{lat}$)
- Outflows ($Q_{ds}$)
- Summed lateral inflows from all reaches within a 50 km upstream walk
  - Add upstream inflows to lateral inflows at any non-headwaters 50 km away

Re-route upstream inflows and lateral inflows through the reach to record:

- Celerity
- Courant number
- X values

---

### Diagnostics

---

#### 1. Is Volume Conserved Across a Reach?

To test that the model conserves volume, inflows to a flowpath are compares to the outflows.  The delta Volume metric is defined as,

```math
\Delta V =
\frac{
\sum_{t=0}^{n} \left(Q_{us}(t) + Q_{lat}(t)\right)
-
\sum_{t=0}^{n} Q_{ds}(t)
}{
\sum_{t=0}^{n} Q_{ds}(t)
}
```

Values close to zero are better than large values. The plots for this metric compare volume conservation across river sizes and reach lengths.  Larger rivers and longer reaches are expected to have worse volume conservation, although these two attributes are often confounding. Generally, if volumetric errors are in the single digits, the model is likely conserving volume well. If a runout period is used, values closer to 1% indicate adequate performance.

##### Distribution of Volume Conservation

![Volume Conservation Distribution](diagnostic_example_images/local_mass_conservation_error.png)

##### Volume Error vs Reach Length

![Volume Error vs Reach Length](diagnostic_example_images/local_mass_conservation_error_vs_dx.png)

---

#### 2. Is Volume Conserved Across the Network?

To test that the model conserves water at the network scale, volume is checked by summing qlats along an upstream walk of the network and compared to the outflow hydrograph.

```math
\Delta V =
\frac{
\sum_{t=0}^{n} \sum_{i=1}^{N} Q_{lat,i}(t)
-
\sum_{t=0}^{n} Q_{ds}(t)
}{
\sum_{t=0}^{n} Q_{ds}(t)
}
```

Values close to zero are better than large values. Longer walks are expected to have worse volume conservation. Generally, if volumetric errors are in the single digits, the model is likely conserving volume well. If a runout period is used, values closer to 1% indicate adequate performance.

##### Distance Walked Upstream vs Volume Lost

![Network Volume Loss vs Distance](diagnostic_example_images/network_mass_conservation_error_vs_walk_dist.png)

---

#### 3. Are Negative Outflows Present?

This simple boolean check should always be false for all reaches.  If it is not, something has gone very wrong. Any negative outflow values indicate a fail.  While there is no plot for this diagnostic, it is summarized in the diagnostics.json.

---

#### 4. Are Attenuation and Hydrograph Acceleration Modest?

Attenuation occurs when a hydrograph peak discharge is reduced across the reach, and acceleration occurs when the peak increases.

```math
\text{Attenuation} =
1 -
\frac{
\max(Q_{ds})
}{
\max(Q_{us} + Q_{lat})
}
```

Generally, attenuation values should be modest across any given reach.  Values in the 0-5% range are typical, but some reaches may go up to ~20%.  Higher values may indicate issues in network discretization, run parameters, or routing code implementation.


##### Histogram of Attenuation Values

![Attenuation Histogram](diagnostic_example_images/attenuation_histogram.png)

---

#### 5. Are Courant Numbers Reasonable?

The courant number describes how fast a flood pulse moves across a reach relative to the reach size and timestep.  Values greater than one means that the flood moves more than one reach length per timestep and generally indicate poor routing performance.

```math
{C}_{n} =
\frac{
{C}_{k}*dx
}{
dt
}
```

##### KDE of Minimum and Maximum Courant Numbers

![Courant KDE](diagnostic_example_images/courant_number_distributions.png)

---

#### 6. Is Lag Proportional to Celerity?

Lag is generally defined as the time from inflow peak t outflow peak. Given that many natural events do not follow the archetypal shape of the unit hydrograph, robust methods for automating lag detection are challenging.  In this metric, lag is calculated as the hydrograph center of mass, and the difference in center of mass between inflow and outflow is used to define lag.  This value is then normalized by length to get an approximate flood pulse translation rate.

```math
t_c = \frac{\sum_{t=0}^{n} t \, Q(t)}{\sum_{t=0}^{n} Q(t)}
```
```math
\Delta t_c = t_{c,ds} - t_{c,us}
```
```math
\text{hydrograph\_translation\_rate}
=
\frac{\Delta x}{\Delta t_c}
```

The flood translation rate is the realization of reach celerity, so it should generally be proportional to the average celerity over the hydrogaph.

##### Celerity vs Centroid Shift Rate

![Celerity vs Centroid Shift](diagnostic_example_images/hydrograph_lag_and_celerity.png)

---

#### 7. How Appropriate Are Reach Lengths?

The combination of reach length and timestep can completely change the results of a Muskingum-Cunge simulation. The "correct" combination has a long history of debate in the scientific literature.  One approach to reach length determination was proposed by Ponce and Theurer in 1982.  It is described below.

ref: V. M. Ponce and F. D. Theurer, “Accuracy Criteria in Diffusion Routing,” Journal of Hydraulics Division, Proceedings of the American Society of Civil Engineers, Vol. 108, No. 6, 1982, pp. 747-757

```math
q_{\text{ref}} = \frac{\max(Q_{in}) + \min(Q_{in})}{2}
```

Let $t^*$ be the index minimizing $|Q_{in}(t) - q_{\text{ref}}|$.

Get reference hydraulic geometry and parameters.

```math
q_{\text{ref}}^{*} = Q_{out}(t^*)
```

```math
c_{\text{ref}} = c(t^*)
```

```math
T_{w,\text{ref}} = T_w(t^*)
```

Calculate ponce optimal dx


```math
\Delta x_{courant} = \Delta t \, c_{\text{ref}}
```

```math
\Delta x_{characteristic} =
\frac{q_{\text{ref}}^{*} / T_{w,\text{ref}}}
{S_0 \, c_{\text{ref}}}
```

```math
\Delta x_{\max} = \frac{1}{2} \left( \Delta x_{courant} + \Delta x_{characteristic} \right)
```

Let

```math
c_{\max} = \max(c)
```

```math
\Delta x_{\min} = c_{\max} \, \Delta t
```

The ideal reach length is

```math
\Delta x_{\text{ideal}} = \max(\Delta x_{\min}, \Delta x_{\max})
```

Finally,

```math
\text{dx\_ratio} = \frac{\Delta x}{\Delta x_{\text{ideal}}}
```

The histogram of this ratio can be used to determine how well the NHF discretization length worked for a given simulation.

##### Histogram + eCDF of $dx$ / Ponce-Optimal Reach Length

![Reach Length Ratio Histogram](diagnostic_example_images/dx_ratio_distribution.png)
