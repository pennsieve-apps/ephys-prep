"""Small helpers shared by the processor: threads, JSON, CSV, timing, memory, heartbeat logging."""
from __future__ import annotations

import csv
import json
import logging
import math
import os
import re
import sys
import threading
import time
from pathlib import Path

log = logging.getLogger("ephys-prep")

SAFE_ID = re.compile(r"[^A-Za-z0-9_-]")


def cgroup_cpus() -> int:
    """CPUs from the cgroup quota (v2 cpu.max, v1 cfs), else the affinity mask; never the host's count alone."""
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()[:2]
        if quota != "max":
            return max(1, math.floor(int(quota) / int(period)))
    except (OSError, ValueError):
        pass
    try:
        q = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
        per = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
        if q > 0 and per > 0:
            return max(1, q // per)
    except (OSError, ValueError):
        pass
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def set_thread_env(threads: int) -> int:
    """Set the BLAS / OpenMP / numba thread counts. Must run before numpy is imported to take full effect."""
    n = threads if threads > 0 else cgroup_cpus()
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = str(n)   # SpikeInterface workers each limit themselves to 1 (max_threads_per_worker)
    if "numba" in sys.modules:
        import numba

        numba.set_num_threads(max(1, min(n, numba.config.NUMBA_NUM_THREADS)))
    else:
        os.environ["NUMBA_NUM_THREADS"] = str(n)
    return n


def cgroup_memory_limit_mb() -> float | None:
    for p in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            v = Path(p).read_text().strip()
            if v != "max" and int(v) < 1 << 60:
                return int(v) / 2**20
        except (OSError, ValueError):
            pass
    return None


def peak_rss_mb() -> float | None:
    """Peak resident memory of the largest single process: this one or its largest finished worker
    (RUSAGE_SELF / RUSAGE_CHILDREN maxima; not the sum over workers, and not the container's cgroup peak,
    which also counts page cache)."""
    try:
        import resource

        unit = 1024 * 1024 if sys.platform == "darwin" else 1024     # bytes on macOS, KiB on Linux
        peak = max(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
        return round(peak / unit, 1)
    except (ImportError, OSError):
        return None


def current_rss_mb() -> float | None:
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return round(int(line.split()[1]) / 1024, 1)
    except OSError:
        pass
    return None


def _clean_json(obj):
    if isinstance(obj, float):
        return None if math.isnan(obj) or math.isinf(obj) else obj
    if isinstance(obj, dict):
        return {str(k): _clean_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_clean_json(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "tolist") and not isinstance(obj, (str, bytes)):
        return _clean_json(obj.tolist())
    if hasattr(obj, "item") and not isinstance(obj, (str, bytes)):
        try:
            return _clean_json(obj.item())
        except (ValueError, AttributeError):
            return str(obj)
    return obj


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(_clean_json(data), fh, indent=2)
        fh.write("\n")


def write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(r.get(k)) for k in columns})


def _fmt(v):
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return "" if math.isnan(v) else f"{v:.6g}"
    return v


class Heartbeat:
    """Logs `what` with elapsed time and memory every `every` seconds until stopped, so long steps never look hung."""

    def __init__(self, what: str, every: float = 60.0, extra=None):
        self.what, self.every, self.extra = what, every, extra
        self._stop = threading.Event()
        self._t0 = time.monotonic()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.wait(self.every):
            msg = f"... {self.what}: {time.monotonic() - self._t0:.0f} s elapsed"
            rss = current_rss_mb()
            if rss is not None:
                msg += f", RSS {rss:.0f} MB"
            if self.extra is not None:
                try:
                    more = self.extra()
                    if more:
                        msg += f", {more}"
                except Exception:   # noqa: BLE001 - a heartbeat must never fail the run
                    pass
            log.info(msg)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=5)
        return False


def worker_count(threads: int, per_worker_mb: float = 1200.0, reserve_mb: float = 1536.0) -> int:
    """Parallel workers: THREADS, but no more than the container's memory allows.

    Each spawned SpikeInterface worker imports the stack (SpikeInterface, PyTorch: ~350 MB) and holds a few 1 s
    chunks of the recording (384 channels: ~50 MB each, float32, several copies in a filter chain), measured at
    up to ~1.2 GB in peak detection + localisation (measured under a 4 GB limit: 3 workers + main went out of
    memory on a 384-channel probe); the main process keeps ~1-1.5 GB (peaks, timestamps, motion).
    """
    limit = cgroup_memory_limit_mb()
    if not limit:
        return threads
    fit = max(1, int((limit - reserve_mb) // per_worker_mb))
    if fit < threads:
        log.info("THREADS=%d but the container has %.0f MB: using %d worker processes (about %.0f MB each)", threads, limit, fit, per_worker_mb)
    return min(threads, fit)


def setup_writable_dirs() -> str:
    """Point every library cache at $TMPDIR (Lambda-safe: the file system is read-only except INPUT_DIR, OUTPUT_DIR,
    HOME and TMPDIR, and nothing baked into the image's /tmp or HOME exists at run time). Returns the scratch dir."""
    import tempfile

    for var in ("HOME", "TMPDIR"):              # the runtime sets them; they may not exist yet
        if os.environ.get(var):
            try:
                os.makedirs(os.environ[var], exist_ok=True)
            except OSError:
                pass
    tmp = os.environ.get("TMPDIR") or tempfile.gettempdir()
    for var, sub in (("MPLCONFIGDIR", "mpl"), ("NUMBA_CACHE_DIR", "numba"), ("XDG_CACHE_HOME", "cache"), ("TRITON_CACHE_DIR", "triton")):
        if not os.environ.get(var):
            os.environ[var] = os.path.join(tmp, sub)
        os.makedirs(os.environ[var], exist_ok=True)
    os.environ.setdefault("USER", "app")           # getpass (torch) with an arbitrary uid and no passwd entry
    return tmp


def process_pool_available() -> bool:
    """Whether worker *processes* can be used: Python's multiprocessing needs POSIX semaphores, which live in
    /dev/shm. Lambda has no /dev/shm (nor does `docker run --ipc none`); there SpikeInterface runs on threads."""
    try:
        import multiprocessing as mp

        mp.get_context("spawn").Lock()
        return True
    except (OSError, ImportError):
        return False


def job_kwargs_for(threads: int) -> dict:
    """SpikeInterface job_kwargs: spawned worker processes (capped by memory) when possible, else threads."""
    if process_pool_available():
        return {"n_jobs": worker_count(threads), "chunk_duration": "1s", "progress_bar": False, "pool_engine": "process",
                "mp_context": "spawn"}
    log.info("no /dev/shm (POSIX semaphores unavailable): SpikeInterface runs on %d threads instead of worker processes", threads)
    return {"n_jobs": threads, "chunk_duration": "1s", "progress_bar": False, "pool_engine": "thread"}


QC_SKIP_DIRS = {"recording", "motion", "sorting", "sorter_output", "kilosort_motion", "_run"}


def copy_small_files(src_root: Path, dst_root: Path) -> None:
    """The `qc` port: every file of the app's top folder except the data folders (recording, motion, sorting,
    sorter_output), keeping the relative paths, so summary.json's figure paths work in both ports."""
    import shutil

    for dirpath, dirnames, filenames in os.walk(src_root):
        dirnames[:] = [d for d in dirnames if d not in QC_SKIP_DIRS]
        rel = Path(dirpath).relative_to(src_root)
        for f in filenames:
            (dst_root / rel).mkdir(parents=True, exist_ok=True)
            shutil.copy2(Path(dirpath) / f, dst_root / rel / f)
