"""Tests for the Muskingum-Cunge Fortran kernel guard against invalid inputs.

The kernel raises a Fortran ``error stop`` when any of (n, s0, z, bw)
is NaN or non-positive. Because ``error stop`` terminates the host
process, each invalid-input case is driven from a child Python
process and we assert on its exit code and stderr.
"""

import subprocess
import sys
import textwrap

import pytest


def _run_kernel(snippet: str) -> subprocess.CompletedProcess:
    """Run a child Python process executing ``snippet`` and return the result.

    The child imports the f2py-compiled Muskingum-Cunge entry point and
    calls it with the parameters defined in ``snippet``. Any ``error stop``
    inside the Fortran kernel terminates the child with a non-zero exit
    code, which is what these tests assert on.
    """
    code = textwrap.dedent(
        """
        from troute.routing.fast_reach.reach import compute_reach_kernel
        {snippet}
        out = compute_reach_kernel(
            dt, qup, quc, qdp, ql, dx, bw, tw, twcc, n, ncc, cs, s0, velp, depthp,
        )
        # If kernel did NOT error_stop, print the result so the parent
        # test can distinguish "guard tripped" from "guard let bad input through".
        print('SURVIVED', out)
        """
    ).format(snippet=snippet)
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
    )


# Baseline parameters that should route cleanly (no guard trip).
_VALID_PARAMS = textwrap.dedent(
    """
    dt     = 300.0
    qup    = 0.5
    quc    = 0.5
    qdp    = 0.5
    ql     = 0.01
    dx     = 1000.0
    bw     = 10.0
    tw     = 20.0
    twcc   = 30.0
    n      = 0.035
    ncc    = 0.05
    cs     = 1.0
    s0     = 0.001
    velp   = 0.5
    depthp = 1.0
    """
)


def test_kernel_accepts_valid_params():
    # Sanity check: the harness is wired up correctly and valid inputs
    # survive the guard.
    result = _run_kernel(_VALID_PARAMS)
    assert result.returncode == 0, (
        f"kernel unexpectedly failed on valid input\n"
        f"stdout: {result.stdout!r}\nstderr: {result.stderr!r}"
    )
    assert "SURVIVED" in result.stdout


_GUARD_MSG = "muskingcungenwm: invalid channel parameter"


@pytest.mark.parametrize(
    "override, label",
    [
        ("s0 = float('nan')", "nan_s0"),
        ("n = float('nan')", "nan_n"),
        ("bw = float('nan')", "nan_bw"),
        ("cs = float('nan')", "nan_cs"),  # z = 1/cs → NaN
        ("s0 = 0.0", "zero_s0"),
        ("s0 = -0.001", "negative_s0"),
        ("n = 0.0", "zero_n"),
        ("n = -0.01", "negative_n"),
        ("bw = 0.0", "zero_bw"),
        ("bw = -1.0", "negative_bw"),
    ],
    ids=lambda v: v if isinstance(v, str) and not v.startswith("(") else None,
)
def test_kernel_error_stops_on_invalid_input(override, label):
    snippet = _VALID_PARAMS + "\n" + override
    result = _run_kernel(snippet)
    # `error stop` terminates with a non-zero exit code.
    assert result.returncode != 0, (
        f"[{label}] kernel was expected to error_stop but survived.\n"
        f"stdout: {result.stdout!r}\nstderr: {result.stderr!r}"
    )
    # The guard message must reach the child stderr/stdout.
    combined = result.stdout + result.stderr
    assert _GUARD_MSG in combined, (
        f"[{label}] expected guard message {_GUARD_MSG!r} in child output.\n"
        f"stdout: {result.stdout!r}\nstderr: {result.stderr!r}"
    )
    # The child must NOT have reached the post-kernel print.
    assert "SURVIVED" not in result.stdout, (
        f"[{label}] kernel did not trip the guard; bad input slipped through."
    )
