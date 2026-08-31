#!/bin/bash
set -e

# This script is called during the ngen Dockerfile if the USE_EWTS
# build argument is ON (or 1, True, or Yes). It enables/uses the
# EWTS already installed in the ngen Dockerfile, then calls the
# compiler.sh to install remaining dependencies

export EWTS_PREFIX="${EWTS_PREFIX:-/opt/ewts}"
export TROUTE_USE_EWTS=ON
export LD_LIBRARY_PATH="${EWTS_PREFIX}/lib:${EWTS_PREFIX}/lib64:${LD_LIBRARY_PATH:-}"

########################################################################
# Change/Verify these values when adopting this script into another org:
#   GH_ORG, EWTS_GIT_REF
########################################################################
# EWTS GitHub source
: "${GH_ORG:=NGWPC}"
: "${EWTS_GIT_REF:=development}"
: "${EWTS_GIT_URL:=https://github.com/${GH_ORG}/nwm-ewts.git}"
: "${EWTS_PY_SUBDIR:=runtime/python/ewts}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Parse only BMI/EWTS-specific args here.
# Pass all standard compiler.sh args through unchanged.
PASSTHROUGH_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gh-org)
            GH_ORG="$2"
            if [[ -z "$2" ]]; then
                echo "Error: --gh-org requires a value"
                exit 1
            fi
            EWTS_GIT_URL="https://github.com/${GH_ORG}/nwm-ewts.git"
            shift 2
            ;;
        --ewts-ref)
            EWTS_GIT_REF="$2"
            if [[ -z "$2" ]]; then
                echo "Error: --ewts-ref requires a value"
                exit 1
            fi
            shift 2
            ;;
        *)
            PASSTHROUGH_ARGS+=("$1")
            shift
            ;;
    esac
done

# Use uv pip if --uv is present in the arguments
USE_UV=false
for arg in "${PASSTHROUGH_ARGS[@]}"; do
    if [[ "$arg" == "--uv" ]]; then
        USE_UV=true
        break
    fi
done

if [[ "$USE_UV" == true ]]; then
    PIP_CMD="uv pip"
    echo "Using uv pip for EWTS installation"
else
    PIP_CMD="pip"
    echo "Using standard pip for EWTS installation"
fi


# When running within the ngen framework, the ngen Dockerfile may
# have already installed the EWTS Python package. Reuse the existing
# installation when available and only install EWTS when it is not
# already present.
if python - <<'PY'
import ewts
import ewts.logger
print(f"Found existing EWTS Python package: {ewts.__file__}")
print(f"Found existing EWTS logger module: {ewts.logger.__file__}")
PY
then
    echo "Using existing EWTS Python package; skipping EWTS install"
else
    echo "EWTS Python package not found; installing from GitHub:"
    echo "  repo: ${EWTS_GIT_URL}"
    echo "  ref:  ${EWTS_GIT_REF}"
    echo "  dir:  ${EWTS_PY_SUBDIR}"

    # Remove any old/stale ewts/troute_ewts from the environment to avoid shadowing
    $PIP_CMD uninstall -y ewts troute_ewts >/dev/null 2>&1 || true

    $PIP_CMD install \
        "ewts @ git+${EWTS_GIT_URL}@${EWTS_GIT_REF}#subdirectory=${EWTS_PY_SUBDIR}"
fi

# Now run the original standalone compiler script for everything else
exec "${SCRIPT_DIR}/compiler.sh" "${PASSTHROUGH_ARGS[@]}"
