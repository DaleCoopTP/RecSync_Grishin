"""One guarded subprocess at a time, with native thread caps set before imports."""
from contextlib import contextmanager
from datetime import datetime, timezone
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .io import read_json, write_json


def configure(threads):
    if threads not in (1, 2):
        raise ValueError("At most two compute threads are allowed")
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = str(threads)
    import pyarrow as pa
    pa.set_cpu_count(threads)
    pa.set_io_thread_count(threads)


@contextmanager
def execution_lock(root):
    import fcntl
    path = Path(root) / "artifacts" / "autofeatures" / ".execution.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another automatic-feature process is already running") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def stop_tree(process):
    # Popen below creates this process group exclusively for our own worker.
    # Cleanup must work even when system-wide PID enumeration is unavailable.
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)


def monitor_process(command, log_path, limit_bytes, cwd=None, env=None):
    """RSS is a conservative sum over the worker and its descendants."""
    import psutil
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    peak = 0
    exceeded = False
    with path.open("a") as stream:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            worker = psutil.Process(process.pid)
            while process.poll() is None:
                rss = 0
                try:
                    for child in [worker, *worker.children(recursive=True)]:
                        try:
                            rss += child.memory_info().rss
                        except psutil.NoSuchProcess:
                            pass
                except psutil.NoSuchProcess:
                    pass
                peak = max(peak, rss)
                if rss > limit_bytes:
                    exceeded = True
                    stop_tree(process)
                    break
                time.sleep(.05)
            return {"returncode": process.wait(), "memory_limit_exceeded": exceeded,
                    "peak_tree_rss_mb": peak / 1024**2, "seconds": time.perf_counter() - started,
                    "rss_accounting": "sum of worker and descendants; shared pages may be counted more than once"}
        except (PermissionError, psutil.AccessDenied) as error:
            stop_tree(process)
            raise RuntimeError("Cannot enforce worker RSS: process inspection is denied; worker stopped") from error
        except BaseException:
            stop_tree(process)
            raise


def guarded_stage(command, config_path, research):
    directory = research.artifacts / "resources"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    log = directory / f"{command}_{stamp}.log"
    args = [sys.executable, "-m", "kion_autofeatures", command, "--config", str(config_path),
            "--name", research.run_name, "--worker"]
    environment = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[key] = str(research.threads)
    environment["PYTHONUNBUFFERED"] = "1"
    print(f"Starting {command}: threads={research.threads}, RSS limit={research.memory_limit_gib} GiB", flush=True)
    result = monitor_process(args, log, int(research.memory_limit_gib * 1024**3), research.root, environment)
    result.update(command=command, log=str(log.relative_to(research.root)), threads=research.threads)
    write_json(directory / f"{command}_{stamp}.json", result)
    if result["memory_limit_exceeded"]:
        raise RuntimeError(f"{command} exceeded RSS limit; completed checkpoints preserved. Log: {log}")
    if result["returncode"]:
        tail = "\n".join(log.read_text().splitlines()[-12:])
        raise RuntimeError(f"{command} failed; completed checkpoints preserved. Log: {log}\n{tail}")
    print(f"Completed {command}: {result['seconds']:.1f}s, peak tree RSS={result['peak_tree_rss_mb']:.1f} MiB; log={log}", flush=True)
    return result
