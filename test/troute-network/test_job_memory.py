"""The memory budget must be the JOB's, not the host's, under any scheduler.

psutil reads /proc/meminfo, which is the host. Under Slurm, PBS, Kubernetes or Docker
the job is confined to far less, and a window sized from the host is sized from memory
the job will never be allowed to touch: the kill arrives at the job's limit.

The trap these tests exist for is that reading the cgroup MOUNT ROOT looks correct in a
container, because a private cgroup namespace puts the limit right there. A Slurm step
is nested several levels down with nothing at the root, so the root read finds no limit
and silently returns the host's free memory.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from troute.job_memory import job_memory_headroom

GB = 1024**3


def _v2(root: Path, rel: str, limit: int | str, current: int = 0,
        inactive_file: int | None = None) -> None:
    d = root.joinpath(*[p for p in rel.strip("/").split("/") if p])
    d.mkdir(parents=True, exist_ok=True)
    (d / "memory.max").write_text(str(limit))
    (d / "memory.current").write_text(str(current))
    if inactive_file is not None:
        (d / "memory.stat").write_text(f"anon 0\ninactive_file {inactive_file}\n")


def _v1(root: Path, rel: str, limit: int, usage: int = 0) -> None:
    d = (root / "memory").joinpath(*[p for p in rel.strip("/").split("/") if p])
    d.mkdir(parents=True, exist_ok=True)
    (d / "memory.limit_in_bytes").write_text(str(limit))
    (d / "memory.usage_in_bytes").write_text(str(usage))


def _proc(tmp_path: Path, text: str) -> Path:
    f = tmp_path / "proc_cgroup"
    f.write_text(text)
    return f


def _call(tmp_path: Path, host_gb: float, proc: Path, resident: int = 0,
          environ: dict[str, str] | None = None):
    return job_memory_headroom(
        int(host_gb * GB), resident,
        cgroup_root=tmp_path / "cgroup", proc_cgroup=proc, environ=environ or {},
    )


class TestSlurmAndPBSNesting:
    """The case the mount-root read misses entirely."""

    SLURM = "0::/system.slice/slurmstepd.scope/job_4242/step_0/user/task_0\n"
    PBS = "0::/pbs_jobs.service/jobid/4242.pbsserver\n"

    def test_a_slurm_step_limit_several_levels_down_is_found(self, tmp_path):
        root = tmp_path / "cgroup"
        _v2(root, "/", "max")                                   # host root: unlimited
        _v2(root, "/system.slice/slurmstepd.scope/job_4242", 12 * GB, 2 * GB)
        budget, source = _call(tmp_path, 128, _proc(tmp_path, self.SLURM))
        assert (budget, source) == (10 * GB, "cgroup")

    def test_a_pbs_cgroup_hook_limit_is_found(self, tmp_path):
        root = tmp_path / "cgroup"
        _v2(root, "/", "max")
        _v2(root, "/pbs_jobs.service/jobid/4242.pbsserver", 8 * GB, 1 * GB)
        budget, source = _call(tmp_path, 128, _proc(tmp_path, self.PBS))
        assert (budget, source) == (7 * GB, "cgroup")

    def test_the_tightest_ancestor_wins_not_the_leaf(self, tmp_path):
        """A limit anywhere up the tree binds, so the effective budget is the min."""
        root = tmp_path / "cgroup"
        _v2(root, "/", "max")
        _v2(root, "/system.slice/slurmstepd.scope", 6 * GB, 0)   # job-level
        _v2(root, "/system.slice/slurmstepd.scope/job_1/step_0", 100 * GB, 0)
        budget, source = _call(tmp_path, 128, _proc(tmp_path,
                               "0::/system.slice/slurmstepd.scope/job_1/step_0\n"))
        assert (budget, source) == (6 * GB, "cgroup")

    def test_the_tightest_wins_when_it_is_the_LEAF(self, tmp_path):
        """The mirror of the case above, and the one that catches "keep the last".

        The walk runs leaf to root, so simply overwriting leaves the ROOT-most limit
        standing. Put a loose but finite limit at the root and the tight one at the
        leaf, and only a real minimum gives the right answer.
        """
        root = tmp_path / "cgroup"
        _v2(root, "/", 100 * GB, 0)                              # finite, but loose
        _v2(root, "/system.slice/slurmstepd.scope/job_9/step_0", 4 * GB, 0)
        budget, source = _call(tmp_path, 128, _proc(tmp_path,
                               "0::/system.slice/slurmstepd.scope/job_9/step_0\n"))
        assert (budget, source) == (4 * GB, "cgroup")

    def test_a_v1_memory_controller_path_is_followed(self, tmp_path):
        root = tmp_path / "cgroup"
        _v1(root, "/slurm/uid_1000/job_42/step_0", 5 * GB, 1 * GB)
        proc = _proc(tmp_path, "9:cpu,memory:/slurm/uid_1000/job_42/step_0\n")
        budget, source = _call(tmp_path, 128, proc)
        assert (budget, source) == (4 * GB, "cgroup")


class TestTheHostStillCounts:
    def test_no_cgroup_and_no_scheduler_falls_back_to_the_host(self, tmp_path):
        (tmp_path / "cgroup").mkdir()
        budget, source = _call(tmp_path, 16, _proc(tmp_path, ""))
        assert (budget, source) == (16 * GB, "host")

    def test_a_host_tighter_than_the_cgroup_wins(self, tmp_path):
        root = tmp_path / "cgroup"
        _v2(root, "/", 64 * GB, 0)
        budget, source = _call(tmp_path, 4, _proc(tmp_path, "0::/\n"))
        assert (budget, source) == (4 * GB, "host")

    def test_a_namespaced_container_still_works_without_procfs(self, tmp_path):
        """Docker and K8s put the container's own limit at the mount root, and
        /proc/self/cgroup may be unreadable. The root must still be read."""
        root = tmp_path / "cgroup"
        _v2(root, "/", 8 * GB, 1 * GB)
        budget, source = _call(tmp_path, 128, tmp_path / "does_not_exist")
        assert (budget, source) == (7 * GB, "cgroup")


class TestTheSchedulerStatesItsOwnAllocation:
    """For setups whose cgroup this process cannot see."""

    def test_slurm_mem_per_node_is_honored(self, tmp_path):
        (tmp_path / "cgroup").mkdir()
        budget, source = _call(tmp_path, 128, _proc(tmp_path, ""),
                               resident=1 * GB,
                               environ={"SLURM_MEM_PER_NODE": str(9 * 1024)})
        assert (budget, source) == (8 * GB, "scheduler")

    def test_slurm_mem_per_cpu_multiplies_by_the_cpu_count(self, tmp_path):
        (tmp_path / "cgroup").mkdir()
        budget, source = _call(tmp_path, 128, _proc(tmp_path, ""),
                               environ={"SLURM_MEM_PER_CPU": "1024",
                                        "SLURM_CPUS_ON_NODE": "6"})
        assert (budget, source) == (6 * GB, "scheduler")

    def test_a_cgroup_tighter_than_the_allocation_wins(self, tmp_path):
        root = tmp_path / "cgroup"
        _v2(root, "/", 2 * GB, 0)
        budget, source = _call(tmp_path, 128, _proc(tmp_path, "0::/\n"),
                               environ={"SLURM_MEM_PER_NODE": str(64 * 1024)})
        assert (budget, source) == (2 * GB, "cgroup")

    @pytest.mark.parametrize("env", [
        {"SLURM_MEM_PER_NODE": "0"},        # Slurm's "all the node's memory"
        {"SLURM_MEM_PER_NODE": "notanumber"},
        {},
    ])
    def test_an_absent_or_unusable_value_is_ignored(self, tmp_path, env):
        (tmp_path / "cgroup").mkdir()
        budget, source = _call(tmp_path, 16, _proc(tmp_path, ""), environ=env)
        assert (budget, source) == (16 * GB, "host")


class TestReclaimableCacheIsNotCountedAsUsed:
    """Usage counts page cache, and this workload streams files through it.

    A long-lived job drifts toward usage == limit while holding gigabytes the kernel
    would return on demand. Reading that as no headroom collapses the window and then
    fails the run outright, on a box with plenty free.
    """

    def test_v2_cache_is_given_back(self, tmp_path):
        root = tmp_path / "cgroup"
        _v2(root, "/", 8 * GB, 8 * GB, inactive_file=6 * GB)   # looks full
        budget, source = _call(tmp_path, 128, _proc(tmp_path, "0::/\n"))
        assert (budget, source) == (6 * GB, "cgroup")

    def test_a_missing_stat_file_is_not_fatal(self, tmp_path):
        root = tmp_path / "cgroup"
        _v2(root, "/", 8 * GB, 2 * GB)
        budget, _ = _call(tmp_path, 128, _proc(tmp_path, "0::/\n"))
        assert budget == 6 * GB


class TestUnlimitedIsNotZero:
    def test_the_v2_word_max_is_not_a_number(self, tmp_path):
        root = tmp_path / "cgroup"
        _v2(root, "/", "max", 0)
        budget, source = _call(tmp_path, 16, _proc(tmp_path, "0::/\n"))
        assert (budget, source) == (16 * GB, "host")

    def test_the_v1_sentinel_is_not_a_limit(self, tmp_path):
        root = tmp_path / "cgroup"
        _v1(root, "/", 2**63 - 4096, 0)
        budget, source = _call(tmp_path, 16, _proc(tmp_path, "9:memory:/\n"))
        assert (budget, source) == (16 * GB, "host")

    def test_the_v1_sentinel_is_skipped_not_merely_outvoted(self, tmp_path):
        """Defensive, and pinned so it stays that way.

        On its own the sentinel is harmless: min() discards 2**63 whatever we do with
        it. It stops mattering only if usage is also large, which is when treating an
        unlimited cgroup as a limited one would invent a small budget out of nothing.
        """
        root = tmp_path / "cgroup"
        _v1(root, "/", 2**63 - 4096, 2**63 - 4096 - 4 * GB)
        budget, source = _call(tmp_path, 16, _proc(tmp_path, "9:memory:/\n"))
        assert (budget, source) == (16 * GB, "host")

    def test_a_cgroup_over_its_limit_reads_as_zero_not_negative(self, tmp_path):
        root = tmp_path / "cgroup"
        _v2(root, "/", 4 * GB, 6 * GB)
        budget, source = _call(tmp_path, 128, _proc(tmp_path, "0::/\n"))
        assert (budget, source) == (0, "cgroup")
