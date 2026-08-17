#!/usr/bin/env bash
# Flag performance or accuracy regressions in your working tree relative to a
# baseline branch (default: development). Builds a t-route image for each side,
# runs the benchmark tiers on both, and compares them with compare_runs.py.
# Exits non-zero if a regression gate fails, suitable as a pre-PR check.
#
# Build model: BOTH sides are built with THIS branch's docker/Dockerfile.dev --
# the Dockerfile your PR merges into development. So the build environment is
# identical on both sides and only the *source code* differs (clean isolation).
# Post-merge that Dockerfile *is* development's, so the baseline is built exactly
# as development will be.
#
# Usage:
#   benchmark/regression_check.sh                  # vs development, Tier A + B
#   benchmark/regression_check.sh --baseline main  # vs another ref
#   benchmark/regression_check.sh --conus          # also Tier C (CONUS data + ~32 GB RAM)
#   benchmark/regression_check.sh --max-slowdown 1.03 --max-rel 1e-9
#
# Prereqs (once): prep the data the harness replays (gitignored, built locally) --
#   python benchmark/prep_ohio_data.py --src /path/to/nhf_1.1.4.gpkg   # Tier A
#   python benchmark/harvest_kernel_inputs.py                          # Tier B (kernel_calls.pkl)
#   python benchmark/prep_conus.py ...                                 # Tier C (only with --conus)
#
# Env overrides:
#   BASELINE_IMG         use a pre-built baseline image, skip the baseline build
#   CANDIDATE_IMG        use a pre-built candidate image, skip the candidate build
#   BASELINE_DOCKERFILE  Dockerfile used for the baseline build (default: this branch's)
#   TROUTE_NATIVE        host arch tuning passed to the builds (default 1)
#   BENCH_COOLDOWN       idle seconds before each side's run (default 0; ~120 for --conus)
#   MAX_SLOWDOWN / MAX_REL / MAX_MEM_GROWTH   regression gates (see compare_runs.py)
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

BASELINE_REF=development
RUN_CONUS=0
NATIVE="${TROUTE_NATIVE:-1}"
COOLDOWN="${BENCH_COOLDOWN:-0}"
MAX_SLOWDOWN="${MAX_SLOWDOWN:-1.05}"
MAX_REL="${MAX_REL:-1e-3}"
MAX_MEM="${MAX_MEM_GROWTH:-1.05}"

while [ $# -gt 0 ]; do
  case "$1" in
    --baseline)     BASELINE_REF="$2"; shift 2;;
    --conus)        RUN_CONUS=1; shift;;
    --native)       NATIVE="$2"; shift 2;;
    --max-slowdown) MAX_SLOWDOWN="$2"; shift 2;;
    --max-rel)      MAX_REL="$2"; shift 2;;
    -h|--help)      sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'; exit 0;;
    *) echo "unknown argument: $1 (try --help)" >&2; exit 2;;
  esac
done

DOCKERFILE="${BASELINE_DOCKERFILE:-$ROOT/docker/Dockerfile.dev}"
GOLDEN_REF="$ROOT/benchmark/.regress_golden"     # baseline output = the accuracy reference
WORKTREE="$(mktemp -u -t troute-regress-XXXXXX)" # git creates it; cleanup removes it
mkdir -p "$ROOT/benchmark/results"

cleanup() {
    git worktree remove --force "$WORKTREE" 2>/dev/null || true
    rm -rf "$WORKTREE" 2>/dev/null || true
}
trap cleanup EXIT

# ---- candidate image: your working tree, this branch's Dockerfile ----
CAND_IMG="${CANDIDATE_IMG:-}"
if [ -z "$CAND_IMG" ]; then
    CAND_IMG=troute-dev:regress-candidate
    echo ">> building candidate image from your working tree ($CAND_IMG)"
    docker build --target dev -f "$ROOT/docker/Dockerfile.dev" \
        --build-arg TROUTE_NATIVE="$NATIVE" -t "$CAND_IMG" "$ROOT"
fi

# ---- baseline image: the ref's source, built with THIS branch's Dockerfile ----
BASE_IMG="${BASELINE_IMG:-}"
if [ -z "$BASE_IMG" ]; then
    SHA="$(git rev-parse --short "$BASELINE_REF")" \
        || { echo "ERROR: baseline ref '$BASELINE_REF' not found" >&2; exit 2; }
    BASE_IMG="troute-dev:regress-baseline-$SHA"
    if docker image inspect "$BASE_IMG" >/dev/null 2>&1; then
        echo ">> reusing baseline image $BASE_IMG ('docker rmi $BASE_IMG' to force a rebuild)"
    else
        echo ">> building baseline ($BASELINE_REF @ $SHA) with $(basename "$DOCKERFILE") -> $BASE_IMG"
        git worktree add --detach "$WORKTREE" "$SHA" >/dev/null
        docker build --target dev -f "$DOCKERFILE" \
            --build-arg TROUTE_NATIVE="$NATIVE" -t "$BASE_IMG" "$WORKTREE"
    fi
fi

# ---- run identical harness on both (the working-tree benchmark/ is bind-mounted,
#      so both images run the same scripts + data; only the compiled code differs) ----
A="python benchmark/bench_e2e.py --runs 5 --warmup 2"
B="python benchmark/bench_kernel.py --runs 15 --warmup 3"
C="python benchmark/bench_conus.py --profile none"
# Baseline captures its own output as the accuracy reference (--save-golden into a
# private dir); the candidate is then compared against it.
BASE_CMD="$A --label regress-base-A --golden-dir benchmark/.regress_golden --save-golden --json && $B --label regress-base-B --json"
CAND_CMD="$A --label regress-cand-A --golden-dir benchmark/.regress_golden --json && $B --label regress-cand-B --json"
if [ "$RUN_CONUS" = 1 ]; then
    BASE_CMD="$BASE_CMD && $C --label regress-base-C --json"
    CAND_CMD="$CAND_CMD && $C --label regress-cand-C --json"
fi

run_side() {  # tag image cmd
    local tag="$1" img="$2" cmd="$3"
    if [ "$COOLDOWN" -gt 0 ]; then
        echo ">> cooldown ${COOLDOWN}s before $tag (thermal reset) ..."; sleep "$COOLDOWN"
    fi
    echo ">> running $tag tiers ($img)"
    docker run --rm -e MALLOC_ARENA_MAX=2 \
        -v "$ROOT/benchmark:/t-route/benchmark" "$img" \
        bash -c "cd /t-route && $cmd"
}

rm -rf "$GOLDEN_REF" 2>/dev/null || true
run_side baseline  "$BASE_IMG" "$BASE_CMD"
run_side candidate "$CAND_IMG" "$CAND_CMD"

# ---- compare + gate ----
echo
CONUS_FLAG=""; [ "$RUN_CONUS" = 1 ] && CONUS_FLAG="--conus"
set +e
python3 benchmark/compare_runs.py \
    --baseline regress-base --candidate regress-cand \
    --max-slowdown "$MAX_SLOWDOWN" --max-rel "$MAX_REL" --max-mem-growth "$MAX_MEM" $CONUS_FLAG
RC=$?
set -e
rm -rf "$GOLDEN_REF" 2>/dev/null || true
exit $RC
