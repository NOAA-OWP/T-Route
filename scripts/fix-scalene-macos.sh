#!/bin/bash
# Strip the Apple-Silicon arm64e slice from scalene's bundled Mach-O
# binaries. Some scalene wheels have shipped fat binaries whose arm64e slice
# is built against the unstable arm64e ABI; loading it on a stock kernel
# aborts with a code-signing/ABI error. The plain arm64 slice works fine, so
# dropping arm64e is a safe, idempotent fix.
#
# No-op on Linux, when scalene is not installed, and when no file has an
# arm64e slice (current scalene wheels need nothing -- this is a safety net
# for future regressions). Run via `pixi run fix-scalene` after installing
# scalene (`pixi run setup-scalene` does both).
set -u

if [ "$(uname -s)" != "Darwin" ]; then
    echo "fix-scalene: non-macOS host, nothing to do"
    exit 0
fi

SCALENE_DIR="$(python -c 'import os, scalene; print(os.path.dirname(scalene.__file__))' 2>/dev/null || true)"
if [ -z "$SCALENE_DIR" ]; then
    echo "fix-scalene: scalene is not installed in this environment, nothing to do"
    exit 0
fi

fixed=0
while IFS= read -r -d '' f; do
    archs="$(lipo -archs "$f" 2>/dev/null || true)"
    case " $archs " in
        *" arm64e "*)
            tmp="$f.noarm64e"
            if lipo "$f" -remove arm64e -output "$tmp" 2>/dev/null; then
                mv "$tmp" "$f"
                echo "fix-scalene: stripped arm64e slice from ${f#"$SCALENE_DIR"/}"
                fixed=$((fixed + 1))
            else
                rm -f "$tmp"
                echo "fix-scalene: WARNING could not strip arm64e from $f" >&2
            fi
            ;;
    esac
done < <(find "$SCALENE_DIR" -type f \( -name '*.so' -o -name '*.dylib' -o -perm +111 \) -print0 2>/dev/null)

if [ "$fixed" -eq 0 ]; then
    echo "fix-scalene: no arm64e slices found, nothing to do"
fi
