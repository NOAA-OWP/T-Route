"""The memory budget this process may actually use, under a scheduler or not.

psutil reads the HOST's free memory. A Slurm or PBS job cgroup is nested well below
the cgroup mount root, so reading the root finds no limit and hands back the host's.
Containers hide this: a private cgroup namespace puts the limit at the root.

RLIMIT_AS is deliberately not consulted. It caps address space, which runs far above
this workload's RSS, so subtracting it would refuse runs that fit.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

__all__ = ["MemoryBudget", "job_memory_headroom"]

# (limit, usage, stat, reclaimable-key) for cgroup v2 and v1.
_V2 = ("memory.max", "memory.current", "memory.stat", "inactive_file")
_V1 = ("memory.limit_in_bytes", "memory.usage_in_bytes", "memory.stat",
       "total_inactive_file")

# v1 writes a number near 2**63 where v2 writes the word "max".
_UNLIMITED = 2**62

MemoryBudget = tuple[int, str]


def _read_int(path: Path) -> int | None:
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def _reclaimable(stat_path: Path, key: str) -> int:
    """Page cache, which usage counts but the kernel reclaims rather than kill for.

    This workload streams forcing and TimeSlice files through it, so a long-lived job
    drifts toward usage == limit while holding gigabytes that are free for the asking.
    """
    try:
        fields = stat_path.read_text().split()
        return int(fields[fields.index(key) + 1])
    except (OSError, ValueError, IndexError):
        return 0


def _own_cgroup(proc_cgroup: Path) -> tuple[str | None, str | None]:
    """This process's v2 and v1 memory paths, from /proc/self/cgroup.

    v2 lines are ``0::/path``; v1 memory lines are ``9:memory:/path``, and the
    controller field may list several together, as in ``9:cpu,memory:/path``.
    """
    v2 = v1 = None
    try:
        lines = proc_cgroup.read_text().splitlines()
    except OSError:
        return None, None
    for line in lines:
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        _, controllers, path = parts
        if controllers == "":
            v2 = path
        elif "memory" in controllers.split(","):
            v1 = path
    return v2, v1


def _tree_headroom(base: Path, rel: str, names: tuple[str, str, str, str]) -> int | None:
    """Smallest remaining budget over this cgroup and its ancestors, or None.

    A limit at ANY level binds, so the headroom is the minimum, not the leaf's.
    """
    limit_name, usage_name, stat_name, cache_key = names
    parts = [p for p in rel.strip("/").split("/") if p]
    best: int | None = None
    for depth in range(len(parts), -1, -1):
        level = base.joinpath(*parts[:depth])
        limit = _read_int(level / limit_name)
        if limit is None or limit >= _UNLIMITED:
            continue                       # no limit here, or v2's literal "max"
        used = _read_int(level / usage_name) or 0
        used -= _reclaimable(level / stat_name, cache_key)
        room = max(0, limit - used)
        best = room if best is None else min(best, room)
    return best


def _scheduler_budget(environ: Mapping[str, str], resident: int) -> int | None:
    """The allocation Slurm says it gave this job, less what is already resident.

    For setups whose cgroup this process cannot see. PBS exports no reliable figure and
    enforces through its cgroup hook, which the tree walk covers.
    """
    total_mb: float | None = None
    per_node = environ.get("SLURM_MEM_PER_NODE")
    per_cpu = environ.get("SLURM_MEM_PER_CPU")
    try:
        if per_node:
            total_mb = float(per_node)
        elif per_cpu:
            cpus = float(environ.get("SLURM_CPUS_ON_NODE") or 1)
            total_mb = float(per_cpu) * cpus
    except ValueError:
        return None
    if not total_mb or total_mb <= 0:
        return None
    return max(0, int(total_mb * 1024 * 1024) - resident)


def job_memory_headroom(
    host_available: int,
    resident: int = 0,
    *,
    cgroup_root: Path = Path("/sys/fs/cgroup"),
    proc_cgroup: Path = Path("/proc/self/cgroup"),
    environ: Mapping[str, str] | None = None,
) -> MemoryBudget:
    """Bytes this process may use, and which of host/cgroup/scheduler set that.

    The name is for the log: an operator seeing a smaller window than expected needs
    to know which constraint decided it.
    """
    env = os.environ if environ is None else environ
    budgets: list[tuple[int, str]] = [(max(0, host_available), "host")]

    v2_path, v1_path = _own_cgroup(proc_cgroup)
    # "/" when /proc/self/cgroup is unreadable: a namespaced container's limit sits
    # at the mount root.
    for rel, base, names in (
        (v2_path if v2_path is not None else "/", cgroup_root, _V2),
        (v1_path if v1_path is not None else "/", cgroup_root / "memory", _V1),
    ):
        room = _tree_headroom(base, rel, names)
        if room is not None:
            budgets.append((room, "cgroup"))

    scheduled = _scheduler_budget(env, resident)
    if scheduled is not None:
        budgets.append((scheduled, "scheduler"))

    return min(budgets, key=lambda pair: pair[0])
