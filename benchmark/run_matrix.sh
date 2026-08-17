#!/usr/bin/env bash
# Build + run the 3-way benchmark matrix, isolating the code win from the
# Python-version win:
#
#   baseline    = pre-PR#94 code (8d17710d), Python 3.9  (no author contribution)
#   after-py39  = optimized code,            Python 3.9  (baseline -> after-py39 = CODE win)
#   after-py311 = optimized code,            Python 3.11 (after-py39 -> after-py311 = PYTHON win)
#
# All three run Tier A/B/C with MALLOC_ARENA_MAX=2; memory is PSS (true
# footprint) from bench_conus.py. A cooldown precedes each arm so all arms
# start from a comparable thermal state; a laptop throttles under sustained
# load, which otherwise penalizes whichever arm runs last. Usage:
#
#   benchmark/run_matrix.sh
#
# Env: TROUTE_NATIVE (default 1), BASELINE_IMG (default troute-dev:baseline),
#      BENCH_COOLDOWN seconds before each arm (default 300).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

NATIVE="${TROUTE_NATIVE:-1}"
BASELINE_IMG="${BASELINE_IMG:-troute-dev:baseline}"
COOLDOWN="${BENCH_COOLDOWN:-300}"

if ! docker image inspect "$BASELINE_IMG" >/dev/null 2>&1; then
    cat >&2 <<EOF
ERROR: baseline image '$BASELINE_IMG' not found. Build it once from the commit
PR #94 was opened against (8d17710d, pre-author-contribution). That source still
has fiona, so use benchmark/Dockerfile.baseline-py39 (py3.9 + gdal-devel):

  git worktree add --detach /tmp/troute-baseline 8d17710d
  docker build --target dev -f benchmark/Dockerfile.baseline-py39 \\
    --build-arg TROUTE_NATIVE=$NATIVE -t $BASELINE_IMG /tmp/troute-baseline
  git worktree remove --force /tmp/troute-baseline
EOF
    exit 1
fi

echo ">> building after-py311 (optimized code, Python 3.11)"
docker build --target dev -f docker/Dockerfile.dev \
    --build-arg TROUTE_NATIVE="$NATIVE" -t troute-dev:after-py311 .

echo ">> building after-py39 (optimized code, Python 3.9)"
docker build --target dev -f benchmark/Dockerfile.py39 \
    --build-arg TROUTE_NATIVE="$NATIVE" -t troute-dev:after-py39 .

run_tiers() {
    local img="$1" tag="$2"
    echo ">> cooling down ${COOLDOWN}s before $tag (thermal reset) ..."
    sleep "$COOLDOWN"
    echo ">> running Tier A/B/C on $tag ($img)"
    docker run --rm \
        -e MALLOC_ARENA_MAX=2 \
        -v "$PWD/benchmark:/t-route/benchmark" \
        "$img" \
        bash -c "cd /t-route && \
            python benchmark/bench_e2e.py    --runs 5  --warmup 2 --label ${tag}-A --json && \
            python benchmark/bench_kernel.py --runs 15 --warmup 3 --label ${tag}-B --json && \
            python benchmark/bench_conus.py  --profile none      --label ${tag}-C --json"
}

run_tiers "$BASELINE_IMG"        baseline
run_tiers troute-dev:after-py39  after-py39
run_tiers troute-dev:after-py311 after-py311

echo
python3 benchmark/summarize_matrix.py baseline after-py39 after-py311
