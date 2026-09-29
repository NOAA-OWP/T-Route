#!/bin/bash
# Find a macOS SDK whose TBD stubs are compatible with the active linker.
# Conda-forge's cctools ld lags behind bleeding-edge macOS SDKs, so the
# default system SDK may not link.  This script probes every installed
# SDK (newest-first, skipping the unversioned MacOSX.sdk symlink) and
# exports SDKROOT / CONDA_BUILD_SYSROOT to the first one that can
# actually produce a linked binary.
#
# Usage (source from a build script):
#   source "$RECIPE_DIR/../../scripts/find_compatible_sdk.sh"
#
# No-op on non-Darwin systems and when every SDK already works.

if [ "$(uname -s)" != "Darwin" ]; then
    return 0 2>/dev/null || exit 0
fi

_sdk_dir="/Library/Developer/CommandLineTools/SDKs"
_probe_src="${TMPDIR:-/tmp}/sdk_probe_$$.c"
echo 'int main(void){return 0;}' > "$_probe_src"
_found_sdk=""

# Sort newest-first so we prefer the most recent compatible SDK.
# Use sort -rV (reverse version sort) to handle e.g. 15.4 > 15 > 14.5.
for _sdk in $(ls -d "$_sdk_dir"/MacOSX*.sdk 2>/dev/null | sort -rV); do
    [ -d "$_sdk" ] || continue
    # Skip the unversioned MacOSX.sdk symlink — it just mirrors the default
    _base="${_sdk##*/}"
    [ "$_base" = "MacOSX.sdk" ] && continue
    if SDKROOT="$_sdk" "${CC:-cc}" -o /dev/null "$_probe_src" 2>/dev/null; then
        _found_sdk="$_sdk"
        break
    fi
done
rm -f "$_probe_src"

if [ -n "$_found_sdk" ]; then
    export SDKROOT="$_found_sdk"
    export CONDA_BUILD_SYSROOT="$_found_sdk"
    echo "SDK probe: using $SDKROOT"
else
    echo "SDK probe: WARNING — no compatible SDK found, keeping SDKROOT=${SDKROOT:-unset}" >&2
fi
