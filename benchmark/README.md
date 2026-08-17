# `benchmark/` t-route performance benchmarks

This directory holds the performance harness for the routing pipeline:
configs, bench drivers, golden output for correctness gates, and the
PNGs that ship with `RESULTS.md`. All measurements are produced inside
the project's DevContainer (`docker/Dockerfile.dev`, Rocky Linux 9)
so they are reproducible on any host that runs Docker.

## What this folder contains

| File | Purpose |
|---|---|
| `RESULTS.md` | Executive summary plus per-change technical writeup with bar charts. **Start here.** |
| `nhf_subset_ohio.yaml` | Tier A config: 1-day, ~11 k flowpaths, single worker. Used for correctness gates and kernel-dominated wall measurement. |
| `conus.yaml` | Tier C config: full CONUS NHF (1.1 M flowpaths), 8 workers, 24 timesteps. Used for production-scale wall measurement. |
| `bench_e2e.py` | Tier A driver. Runs the full `nwm_routing` CLI, measures wall/CPU/RSS (resident set size), compares output to `golden/` (PASS / FAIL gate). |
| `bench_conus.py` | Tier C driver. Single CONUS run with `--profile {cprofile,pyspy,none}` for hot-path analysis. |
| `bench_kernel.py` | Tier B microbenchmark. Replays harvested `compute_network_structured()` calls so the MC kernel can be timed in isolation, without the Python pipeline around it. |
| `harvest_kernel_inputs.py` | Records the kernel inputs from a real Tier A run into `data/kernel_calls.pkl`, so the kernel bench can replay them deterministically. |
| `regression_check.sh` | **Pre-PR regression check.** Builds your working tree and a baseline ref (default `development`), runs the tiers on both, and flags performance/accuracy regressions. Exits non-zero on failure. See "Checking your changes for regressions" below. |
| `compare_runs.py` | The baseline-vs-candidate comparison + gating step behind `regression_check.sh`; also runnable by hand on any two result tags. |
| `run_matrix.sh`, `summarize_matrix.py` | Build and run the three-way study matrix (baseline / after-py39 / after-py311) and print the code-vs-Python attribution table behind `RESULTS.md`. |
| `prep_ohio_data.py`, `prep_conus.py` | Build the Tier A and Tier C input data from the NHF v1.1.4 CONUS GeoPackage. |
| `run_conus_lakes.py` | **NHF release validation.** Routes a CONUS geopackage as-is (read-only, lakes kept) and verifies finite output; the waterbody warnings during the network build are the per-release lake census. Exit code is CI-friendly. |
| `sweep_max_loop_size.py` | Runs Tier A across a sweep of `max_loop_size` values, captures wall/CPU/RSS per point, writes `results/max_loop_size_sweep.json`. Backs the operational deployment recommendation on chunk sizing. |
| `plot_max_loop_size.py` | Renders the sweep JSON to `figures/max_loop_size_sweep.png`. |
| `data/`, `golden/` | Input GeoPackages and reference output netCDFs (gitignored; build locally). |
| `results/` | Per-run JSON metric files (`{label}.json`, `{label}.kernel.json`, `{label}.conus.json`, `max_loop_size_sweep.json`). |
| `figures/` | PNG bar charts embedded in `RESULTS.md`. Regenerate via `python benchmark/generate_figures.py`. |
| `generate_figures.py` | Builds the bar charts in `figures/` from the measured numbers (constants at the top). |

## High-level summary

![Overall improvement across all three tiers](figures/speedup_overview.png)

The contribution drops **CONUS wall time from 275.8 s to 115.3 s
(2.39x speedup)** while reducing CPU time by 1.66x and pushing worker
utilization from 1.40x to 2.02x of 8 cores. Tier A wall improves 1.18x
and the isolated MC-kernel replay (Tier B) improves 1.13x. Memory is
essentially flat (~1.08x; true footprint ~28-30 GB measured as PSS,
proportional set size, not an inflated per-process RSS sum). Output is bit-identical to a golden
saved with the optimized build on the correctness gate. Numbers are
DevContainer measurements (Python 3.11, `MALLOC_ARENA_MAX=2`) from the
cooldown-gated three-way matrix. See `RESULTS.md` for the baseline
(pre-#94), code, and Python 3.11 breakdown.

The work is grouped into four tracks:

1. **Toolchain and build environment** (`docker/Dockerfile.dev`,
   `pyproject.toml`): **Python 3.9 -> 3.11** (the production target),
   picking up CPython's interpreter speedups (~6% on CONUS, in the
   Python-heavy graph-construction phase); `fiona` dropped for
   `pyogrio` (no system GDAL or C++ toolchain); system packages in one
   cache-cleaned layer; `ccache` + BuildKit cache mounts for rebuilds;
   image ~1.82 -> 1.4 GB.
2. **Kernel-level** (`src/kernel/muskingum/`):
   `-O3 -funroll-loops` build (with optional `TROUTE_NATIVE=1`
   for host-specific `-mcpu=native`/`-march=native` tuning); hoisted
   loop-invariant transcendentals; strength-reduced powers; common
   subexpression elimination (CSE) on the upstream-weighted sum.
3. **Routing-side** (`src/troute-routing/troute/routing/compute.py`):
   eliminated per-cluster `deepcopy`; consolidated 6+ per-cluster
   `.reindex` calls into one extended-index `pd.api.extensions.take`;
   per-cluster fast-path guards; `.to_numpy(copy=False)` migration.
4. **Graph construction** (`src/troute-network/troute/`): vectorized
   `_discretize_links`, `extract_connections`, and the two
   `groupby.apply(list).to_dict()` calls in
   `crosswalk_nex_flowpath_poi`.

## Reproducing the results

Everything runs inside the DevContainer. The compiled core (Fortran
plus Cython extensions) is built by `compiler.sh` during the Docker
image build.

### Build the DevContainer image

```bash
docker build --target dev -f docker/Dockerfile.dev \
  --build-arg TROUTE_NATIVE=1 \
  -t troute-dev:bench .
```

`TROUTE_NATIVE=1` enables host-specific `-mcpu=native` /
`-march=native` arch tuning on the MC Fortran kernel. The
benchmark numbers in `RESULTS.md` were taken with this flag.
Omit it (the project default) for a portable build safe to run
on a different CPU than the build host, the right choice for
shipping container images or conda packages across heterogeneous
clusters, at a small wall-time cost on the kernel.

The build produces an image with the t-route source compiled in
place under `/t-route` and the Python venv at `/opt/venv` already
on `PATH`. The bench commands below assume you launch a container
from that image with the bind mounts shown.

The dev image builds on Python 3.11 (the production target) and is
about 1.4 GB. It needs no system GDAL or C++ toolchain: geopandas reads
geopackages through `pyogrio`, whose manylinux aarch64 wheels bundle
GDAL. The headline numbers in this folder are measured on Python 3.11;
the benchmark matrix also runs the Python 3.9 arm to isolate the
interpreter-version effect, and the optimized build reproduces the
Tier A output bit-for-bit on both.

### Memory requirements

CONUS (Tier C) peaks at **~25 GB resident** in the main process
and **~28 GB across the whole process tree** (main + 8 `joblib`
workers, measured as PSS with `MALLOC_ARENA_MAX=2`). **Configure
your container runtime with at least ~32 GB of RAM available to
the VM.** Without this the output-write step will be OOM-killed
(exit code -9) and you'll see the routing log stop mid-way at
"Handling output ..."

How you bump the VM's RAM ceiling depends on your runtime:

- **Docker Desktop:** Settings -> Resources -> Memory slider.
- **Podman:** `podman machine set --memory 32768` then
  `podman machine restart`.
- **OrbStack:** `orb config set memory_mib 32768 && orb stop && orb start`.
- **colima:** `colima start --memory 32`.

Tiers A and B run comfortably in the default 8-16 GB; only Tier C
needs the bump.

### Reducing memory with `max_loop_size`

`max_loop_size` chunks the forcing file list, trading peak memory
against wall time: smaller chunks cut the transient footprint but add
per-chunk overhead. A Tier A sweep (`nhf_subset_ohio`):

![max_loop_size sweep on Tier A: wall/CPU and peak RSS vs chunk size](figures/max_loop_size_sweep.png)

Peak RSS grows steeply with chunk size (about 415 MB at `mls=1` to about
8.8 GB at `mls=144`) while wall time plateaus past `mls=24`, so a
moderate value captures most of the speed at a fraction of the memory.
See "Operational deployment recommendations" in `RESULTS.md` for the
full table and guidance.

### Memory measurement (why `MALLOC_ARENA_MAX=2`)

Without an arena cap, glibc's ptmalloc2 allocator reserves several
MB of virtual memory per thread arena (capped at `8 * ncores`),
which inflates baseline RSS by tens of GB before any application
allocation happens. That overhead is identical in baseline and
after, so it doesn't affect *speedup ratios*, but it swamps the
absolute RSS numbers enough to hide the optimization-driven
changes.

Setting `MALLOC_ARENA_MAX=2` caps the arena count at 2 per
process, trading negligible allocator contention (the main
process is effectively single-threaded; `joblib` workers are
separate processes) for honest peak-memory measurements. This is a
common production setting in Python data services (Airflow, Dask,
etc.). Omitting it does not materially change the wall-time ratios,
but it inflates the absolute memory numbers (a naive RSS sum can
exceed physical RAM), hiding the true ~28 GB footprint reported in
`RESULTS.md`.

### Source data

Both tiers derive from the **NextGen Hydrofabric v1.1.4 CONUS
GeoPackage** (`nhf_1.1.4.gpkg`, ~6 GB). The numbers in `RESULTS.md`
were produced against this exact dataset; reviewers with access to
it can reproduce them directly. Remember the local path; the
remaining steps both consume that single file.

### Set up the input data (one-time)

Run the prep scripts inside a container. The host path holding
`nhf_1.1.4.gpkg` is bind-mounted read-only at `/hydrofabric`, and
the benchmark directory is bind-mounted so the synthesized inputs
land in your working tree (the image itself does not bake
`benchmark/` in):

```bash
docker run --rm \
  -e MALLOC_ARENA_MAX=2 \
  -v /path/to/hydrofabric:/hydrofabric:ro \
  -v "$(pwd)/benchmark:/t-route/benchmark" \
  troute-dev:bench \
  bash -c "cd /t-route && \
    python benchmark/prep_ohio_data.py  --src /hydrofabric/nhf_1.1.4.gpkg && \
    python benchmark/prep_conus.py --src /hydrofabric/nhf_1.1.4.gpkg && \
    python benchmark/bench_e2e.py --save-golden && \
    python benchmark/harvest_kernel_inputs.py"
```

What each step does:

- `prep_ohio_data.py` carves the upstream subgraph of `fp_id` 1725641 (Ohio
  River basin, 11,327 flowpaths) and synthesizes 144 hourly forcing
  CSVs. Output: `benchmark/data/domain/nhf_subset_ohio.gpkg`.
- `prep_conus.py` processes the full CONUS hydrofabric and
  synthesizes 2 hourly forcing CSVs. Output:
  `benchmark/data/conus/nhf_conus.gpkg`.
- `bench_e2e.py --save-golden` runs one Tier A invocation and saves
  its output netCDFs under `benchmark/golden/` as the correctness
  reference. Every later run compares against this.
- `harvest_kernel_inputs.py` intercepts the
  `compute_network_structured` calls from a Tier A run and pickles
  their arguments into `benchmark/data/kernel_calls.pkl`, so Tier B
  can replay the MC kernel without the Python pipeline.

Both prep scripts **synthesize forcing** (constant per-segment
`q_lateral`), so no external CHRTOUT or forcing dataset is needed.
The MC kernel is forcing-shape-sensitive, not forcing-value-
sensitive, for performance purposes: identical shapes produce
representative timing. The saved golden becomes your local
correctness reference; subsequent runs PASS if they reproduce that
output bit-for-bit, FAIL on new NaN/Inf.

### Run the benchmarks

Once the prep step has completed (golden and kernel inputs are now
on disk under `benchmark/`), run the three tiers inside containers
with the same bind mount. `MALLOC_ARENA_MAX=2` is set so glibc
arena overhead doesn't dominate the RSS measurements (see "Memory
measurement" below):

```bash
docker run --rm \
  -e MALLOC_ARENA_MAX=2 \
  -v "$(pwd)/benchmark:/t-route/benchmark" \
  troute-dev:bench \
  bash -c "cd /t-route && \
    python benchmark/bench_e2e.py    --runs 5  --warmup 2 --label tierA --json && \
    python benchmark/bench_kernel.py --runs 15 --warmup 3 --label tierB --json && \
    python benchmark/bench_conus.py  --profile none      --label tierC --json"
```

Per-tier notes:

- **Tier A** (`bench_e2e.py`) reports wall/CPU/RSS over the 5 timed
  runs, compares output to `golden/`, prints `PASS` / `FAIL`. Save a
  fresh golden any time via `--save-golden`.
- **Tier B** (`bench_kernel.py`) replays the harvested
  `compute_network_structured()` calls only: no I/O, no `joblib`, no
  config parsing. Use this to measure pure MC-kernel changes.
- **Tier C** (`bench_conus.py`) does a single CONUS run. Swap
  `--profile none` for `--profile cprofile` to write
  `results/<label>.conus.pstats`, or `--profile pyspy` for an SVG
  flamegraph (the latter requires the container to have `--privileged`
  or `--cap-add SYS_PTRACE`).

### Regenerating the bar charts in `RESULTS.md`

```bash
docker run --rm \
  -v "$(pwd)/benchmark:/t-route/benchmark" \
  troute-dev:bench \
  python /t-route/benchmark/generate_figures.py
```

The script reads its numbers from constants at the top of the file
(the `CONUS`, `CONUS_PHASES`, `TIER_A`, `TIER_B` blocks); update
those, rerun, and commit the new PNGs in `figures/`.

### VS Code Dev Containers workflow

The repo's `.devcontainer/devcontainer.json` resolves the
`/hydrofabric` bind-mount source from the host environment
variable `TROUTE_HYDROFABRIC_DIR`, falling back to
`/tmp/troute-no-hydrofabric` (which the DevContainer's
`initializeCommand` creates on the host before container start)
so the container starts cleanly for non-benchmark users.

To run benchmarks from inside VS Code:

1. Export the env var in your shell **before** launching VS Code so
   it gets inherited:
   ```bash
   export TROUTE_HYDROFABRIC_DIR=/absolute/path/to/your/hydrofabric/dir
   code /path/to/t-route
   ```
2. "Reopen in Container" picks up the mount automatically; the
   GeoPackage is then visible at `/hydrofabric` inside the
   container.
3. Set `MALLOC_ARENA_MAX=2` in your shell or in the `containerEnv`
   block of `devcontainer.json` so every terminal in the container
   inherits it.
4. From an integrated terminal the bench commands are the same as
   above except you drop the outer `docker run` shell, since you
   are already inside the container.

If `TROUTE_HYDROFABRIC_DIR` is unset, the container still starts
with `/hydrofabric` mapped to the empty `/tmp/troute-no-hydrofabric`
directory; only the benchmark prep scripts will complain.

## Validating a new NHF release (lakes included)

The performance tiers above intentionally empty the lakes layer; to check
whether a new NextGen Hydrofabric release still routes through t-route with
waterbodies enabled, run:

```bash
python benchmark/run_conus_lakes.py --src /path/to/nhf_conus.gpkg
```

The script reads the geopackage read-only (no copy, no data repair), builds
the full network with `break_network_at_waterbodies: True`, routes a short
window, and exits non-zero if the run crashes, produces non-finite output, or
the data fails the channel-parameter pre-checks. The warnings emitted during
the network build report, per category and with counts, every lake that could
not be modeled as a reservoir (non-numeric lake_id, no fp_id, missing
level-pool parameters, no single inlet -> outlet chain) -- treat them as the
release's lake census. Needs ~16 GB RAM and a few minutes.

## Checking your changes for regressions

Before opening a PR, compare your working tree against the branch you
are merging into (default `development`) to catch performance or
accuracy regressions. One script does the whole thing:

```bash
# one-time: build the data the harness replays (see "Set up the input data" above)
python benchmark/prep_ohio_data.py --src /path/to/nhf_1.2.1.gpkg
python benchmark/harvest_kernel_inputs.py

# compare your working tree vs development (Tier A + Tier B)
benchmark/regression_check.sh

# also include the CONUS tier (needs the CONUS dataset + ~32 GB RAM)
benchmark/regression_check.sh --conus
```

`regression_check.sh` builds a Docker image for each side, runs the
benchmark tiers on both, captures the baseline's own output as the
accuracy reference, compares the two, and prints a verdict table:

```text
  check                    baseline             candidate    speedup  verdict
  -------------------  ------------  --------------------  ---------  -------------
  Tier A wall               54.83 s               46.30 s      1.18x  ok (faster)
  Tier B kernel          3161.18 ms            2787.56 ms      1.13x  ok (faster)
  Accuracy (flow rel)     reference       0.0e+00 (NaN 0)          -  ok (identical)

OK: no regression beyond the gates
```

It **exits non-zero if any gate fails**, so it drops into a pre-PR
hook or CI step unchanged. The gates (override on the command line or
via env var):

| gate | default | meaning |
|---|---|---|
| `--max-slowdown` | `1.05` | fail if the candidate is more than 5% slower on any tier |
| `--max-rel` | `1e-3` | fail if Tier A **flow** output drifts more than this vs the baseline |
| `MAX_MEM_GROWTH` | `1.05` | (with `--conus`) fail if candidate peak PSS grows more than 5% |

**Accuracy** is measured against the baseline's *own* output, captured
fresh each run (not a committed golden), so a pure refactor reads
`0.0e+00 (NaN 0)  ok (identical)`. Any **new NaN/Inf** always fails.
The relative-drift gate is on **flow** only, the conserved routing
output; velocity and depth relative error blows up at near-dry nodes
even between identical builds, so they are reported for information and
gated only on new NaN. If your change *intends* to alter results, the
gate flags it for review; confirm the new numbers are correct, then
loosen `--max-rel`.

**Build model:** both sides are built with **this branch's**
`docker/Dockerfile.dev` (the Dockerfile your PR merges into
development), so the build environment is identical on both sides and
only the source code differs. Post-merge that Dockerfile is exactly
development's, so the baseline is built as development will be.

Useful overrides:

- `--baseline <ref>`: compare against a ref other than `development`.
- `BASELINE_IMG` / `CANDIDATE_IMG`: reuse a pre-built image and skip
  that build (fast iteration, or to supply a known-good baseline image
  when an older `development` predates the current Dockerfile).
- `BENCH_COOLDOWN=120`: idle before each side so a throttling laptop
  doesn't penalize whichever side runs second (recommended with
  `--conus`).

Single-laptop wall times are noisy and thermal-sensitive: treat a
borderline `slower` verdict as "re-run to confirm," and prefer a quiet
machine (or `BENCH_COOLDOWN`) for the CONUS tier. The accuracy gate is
deterministic.

### Comparing two existing runs by hand

`compare_runs.py` is the pure comparison step and works on any two
result tags already in `benchmark/results/`:

```bash
python benchmark/compare_runs.py \
  --baseline regress-base --candidate regress-cand --conus
```

## Notes

- All measurements were taken inside the DevContainer on linux/arm64.
  Relative speedups generalize; absolute seconds will shift with the
  host CPU and the container runtime's I/O overhead.
- `cpu_pool=1` in `nhf_subset_ohio.yaml` and `cpu_pool=8` in
  `conus.yaml` is deliberate. Tier A measures kernel-dominated
  single-thread cost; Tier C measures whether the workers are
  well-fed.
- cProfile inflates Python-loop self-time by 2-5x. Use clean runs
  (`--profile none`) for headline numbers; reserve cProfile for
  attribution.
