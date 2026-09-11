"""One window partition, shared by the CLI and the BMI.

They had drifted: the CLI filled to ``max_loop_size`` and folded a short final
remainder into its neighbor, the BMI split evenly, so 100 columns at a width of 24 came
out ``[24, 24, 24, 28]`` on one and ``[20] * 5`` on the other. The even split is the one
that survives both constraints, since a folded window is wider than the caller sized for.
"""

from __future__ import annotations

import math

__all__ = ["AUTO_WINDOW", "plan_windows", "resolve_window"]

# Forcing columns per window when max_loop_size is 0 (automatic). benchmark/RESULTS.md
# plateaus in wall time here while peak RSS keeps climbing, so wider buys no speed.
AUTO_WINDOW = 24

WindowBounds = list[tuple[int, int]]


def plan_windows(n_columns: int, window: int, span: int = 0) -> WindowBounds:
    """Half-open ``(start, stop)`` bounds covering ``n_columns`` forcing columns.

    Widths differ by at most one, so none exceeds the ceiling of the average. Every
    window also reaches ``span``, the DA's horizon in columns, whenever any split can:
    a window below it reads across a boundary it cannot supply. Callers with a memory
    ceiling check the widest bound against it; this has no opinion about memory.
    """
    if n_columns <= 0:
        return []
    width = max(1, min(window, n_columns))
    count = math.ceil(n_columns / width)
    if span > 0 and count > 1 and n_columns // count < span:
        # Give up the CAP minimally, not the split. Collapsing to one window turned a
        # 2161-column run at span 60 into a single 2161-column window, ~120 GB, where
        # 36 windows of 60 and 61 hold the span. The caller's memory check gates the
        # roughly one column of excess.
        count = max(1, n_columns // span)
    base, extra = divmod(n_columns, count)
    bounds: WindowBounds = []
    start = 0
    for i in range(count):
        stop = start + base + (1 if i < extra else 0)
        bounds.append((start, stop))
        start = stop
    return bounds


def resolve_window(
    configured: int | None, span: int = 0, output_cols: int = 0
) -> int:
    """Columns per window from the config alone, identically for either driver.

    ``configured`` of 0 or None is the schema's "automatic". ``span`` and
    ``output_cols`` both raise it: a window under the DA's horizon changes discharge,
    and one window writes one output part. The output cadence belongs here rather than
    in one driver because it moves the routing partition, and the CLI enlarged for it
    while the BMI did not. Only the memory cap is a driver's own, since only the BMI
    measures memory.
    """
    return max(int(configured or AUTO_WINDOW), span, output_cols, 1)
