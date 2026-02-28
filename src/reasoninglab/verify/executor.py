"""
verify/executor.py
──────────────────
Runs model-generated candidate code against a deterministic test suite in a
isolated subprocess and returns a structured result.

Isolation layers (applied in order):
  1. Subprocess       – separate process; runner memory/state is unreachable.
  2. Safety preamble  – nullifies destructive OS/network/subprocess calls
                        before any model code executes (_SAFETY_PREAMBLE).
  3. Temp working dir – cwd=tempdir so any files the script creates are
                        contained and deleted automatically.
  4. Wall-clock timeout – subprocess.run(timeout=) kills the process if it
                          runs longer than timeout_s seconds.

Known limitations (documented, not silently accepted):
  - No hard memory limit on Windows.  resource.setrlimit is Linux-only and
    there is no pywin32-free equivalent.  A memory bomb will be killed by the
    Windows OOM manager eventually, but the elapsed_s measurement for that
    attempt will be unreliable.
  - ctypes can bypass Python-level monkey-patching via direct syscalls.
    Not a realistic concern for model-generated algorithm code.
  - The preamble is NOT a security sandbox (same disclaimer as OpenAI
    HumanEval).  It prevents accidental destructive calls, not deliberate
    attacks.

Pass detection uses a sentinel protocol (not just returncode == 0):
  passed = returncode == 0  AND  _PASS_SENTINEL present in stdout.
  This guards against early-exit bypasses (raise SystemExit(0), os._exit(0))
  where the process exits cleanly but tests never ran.

Design mirrors OpenAI HumanEval's reliability_guard() approach, adapted for
Windows (no resource.setrlimit, no signal.SIGALRM, subprocess.run timeout
replaces the signal-based time_limit() context manager).
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Safety preamble
# ─────────────────────────────────────────────────────────────────────────────
# This string is prepended to every candidate script before it is written to
# disk and executed.  It runs as ordinary Python inside the subprocess, before
# any model-generated code, so all dangerous names are nullified first.
#
# Structure mirrors OpenAI HumanEval's reliability_guard(); differences:
#   - resource.setrlimit removed  (Linux-only; no Windows equivalent)
#   - signal.SIGALRM removed      (Linux-only; timeout handled by subprocess)
#   - sys.exit = None added       (critical for subprocess correctness, see below)
#   - socket.socket = None added  (network blocking; HumanEval omits this)
#   - os.listdir / os.scandir blocked (read-only but unnecessary for algorithms)
#   - Linux-only os.* attributes wrapped in hasattr() for portability

_SAFETY_PREAMBLE = '''\
# ── ReasoningLab safety preamble ─────────────────────────────────────────────
# Runs before any model-generated code.
# WARNING: this is NOT a security sandbox.  It prevents accidental destructive
# calls from model-generated algorithm code; it does not protect against
# deliberate bypass via ctypes or importlib.reload().
# ─────────────────────────────────────────────────────────────────────────────

import builtins as _builtins
import faulthandler as _faulthandler
import os as _os
import shutil as _shutil
import socket as _socket
import subprocess as _sp
import sys as _sys

# ── Disable Python crash reporter ─────────────────────────────────────────────
# faulthandler writes extra text to stderr on hard crashes (segfault, etc.).
# We read stderr in taxonomy.py to classify failure types; unexpected noise
# from faulthandler would corrupt that classification.
_faulthandler.disable()

# ── Prevent fake "passed" results ─────────────────────────────────────────────
# If model code exits with code 0 BEFORE the tests run, our executor would
# incorrectly report "passed" → silently corrupted experiment data.
#
# Four distinct early-exit mechanisms exist; we handle each differently:
#
#   exit() / quit()     → Python builtins    → blocked by setting to None below.
#   sys.exit(0)         → function call      → blocked by setting to None below.
#   os._exit(0)         → C-level immediate  → blocked by setting to None below.
#   raise SystemExit(0) → direct exception   → cannot be blocked (uses the
#                           exception class directly, not a function call);
#                           handled by the sentinel protocol instead.
#
# The sentinel (see _build_script / execute_candidate) is the definitive guard:
# passed=True only when returncode==0 AND the sentinel token appears in stdout,
# proving that execution reached the end of the script.
_builtins.exit = None
_builtins.quit = None
_sys.exit      = None
_os._exit      = None   # C-level exit; bypasses Python exception handling

# ── Reproducibility: cap NumPy / SciPy thread count ──────────────────────────
# NumPy uses OpenMP internally and by default occupies ALL CPU cores for matrix
# operations.  That makes latency measurements unstable across runs (core count
# varies with OS scheduling).  Capping at 1 thread makes runtimes comparable.
_os.environ["OMP_NUM_THREADS"] = "1"

# ── Process control ───────────────────────────────────────────────────────────
_os.kill   = None   # cannot send signals to arbitrary PIDs on the machine
_os.system = None   # cannot run shell commands (e.g. os.system("del /f ..."))
_os.putenv = None   # cannot set C-level env vars (affects child processes)

# ── File deletion ─────────────────────────────────────────────────────────────
# All four names refer to deletion; os.remove and os.unlink are equivalent
# (Unix heritage – two names for the same syscall).
_os.remove      = None   # delete a file
_os.unlink      = None   # same as remove
_os.removedirs  = None   # delete a chain of directories
_os.rmdir       = None   # delete a single empty directory
_shutil.rmtree  = None   # recursively delete an entire directory tree (most dangerous)

# ── File modification ─────────────────────────────────────────────────────────
_os.rename    = None   # rename / move a file
_os.renames   = None   # rename with automatic intermediate directory creation
_os.replace   = None   # like rename but atomically overwrites the destination
_os.truncate  = None   # shrink a file to N bytes (corrupts without deleting)
_shutil.move  = None   # high-level move (copy + delete)

# ── Directory navigation ──────────────────────────────────────────────────────
# We set cwd=tempdir in subprocess.run(); blocking chdir prevents model code
# from navigating out of that temp directory into the actual project tree.
# Blocking getcwd removes the ability to discover the current path, making
# any attempted navigation blind.
_os.chdir  = None
_os.getcwd = None

# ── Directory listing ─────────────────────────────────────────────────────────
# listdir / scandir can explore ANY directory the current user can read, not
# just the current folder.  Algorithm tasks have no legitimate need for this.
_os.listdir = None
_os.scandir = None

# ── File permissions ──────────────────────────────────────────────────────────
# chmod can make files world-writable or executable; chown transfers ownership.
_os.chmod = None
if hasattr(_shutil, "chown"):
    _shutil.chown = None   # shutil.chown is Unix-only; hasattr guards Windows

# ── Subprocess / process spawning ─────────────────────────────────────────────
# subprocess.Popen is the base class; all other subprocess functions call it
# internally.  Blocking Popen alone would be sufficient, but we block the
# convenience wrappers explicitly for clarity.
_sp.Popen             = None
_sp.run               = None
_sp.call              = None
_sp.check_call        = None
_sp.check_output      = None
_sp.getoutput         = None
_sp.getstatusoutput   = None

# ── Network ───────────────────────────────────────────────────────────────────
# socket.socket is the constructor for ALL network connections (TCP, UDP, ...).
# Nullifying it prevents outbound connections (HTTP requests, data exfiltration,
# etc.).  Algorithm tasks have no legitimate reason to open a socket.
# Note: HumanEval does not block this; we add it as an extra layer.
_socket.socket = None

# ── Block dangerous third-party imports ───────────────────────────────────────
# Setting sys.modules[name] = None makes "import <name>" raise ImportError.
#
#   joblib   – spawns parallel worker processes, bypassing subprocess isolation
#   psutil   – can list, inspect, and kill any process on the machine
#   resource – on Linux could remove memory/CPU limits (irrelevant on Windows,
#              but blocked for portability)
#   ipdb     – interactive debugger; hangs waiting for terminal input
#   tkinter  – GUI toolkit; hangs looking for a display in a subprocess
_sys.modules["joblib"]   = None
_sys.modules["psutil"]   = None
_sys.modules["resource"] = None
_sys.modules["ipdb"]     = None
_sys.modules["tkinter"]  = None

# ── Linux-only os attributes ──────────────────────────────────────────────────
# These functions do not exist on Windows, so hasattr() returns False and the
# block is silently skipped.  They are included so the preamble also works
# correctly on Linux (where this code might run in the future).
#
# On Windows the corresponding risks are covered differently:
#   fork / forkpty  → fork bomb risk → covered by blocking subprocess.Popen
#   killpg          → kill process group → covered by blocking os.kill
#   setuid / setgid → privilege escalation → not possible in Python on Windows
#   fchdir / fchmod / fchown / chroot / lch* → file-descriptor or Unix-only
#     filesystem ops → do not exist on Windows, no equivalent hole to cover
_LINUX_ONLY_OS_ATTRS = [
    "fork",       # clone the current process (fork bomb risk)
    "forkpty",    # fork with a pseudo-terminal
    "killpg",     # send a signal to an entire process group
    "setuid",     # change process user ID (privilege escalation)
    "setgid",     # change process group ID
    "seteuid",    # change effective user ID
    "setegid",    # change effective group ID
    "fchdir",     # change directory via file descriptor
    "fchmod",     # change file permissions via file descriptor
    "fchown",     # change file ownership via file descriptor
    "chown",      # change file ownership (not available on Windows)
    "chroot",     # change filesystem root (serious privilege escalation)
    "lchflags",   # change flags on a symbolic link itself
    "lchmod",     # change permissions on a symbolic link itself
    "lchown",     # change ownership on a symbolic link itself
]
for _attr in _LINUX_ONLY_OS_ATTRS:
    if hasattr(_os, _attr):
        setattr(_os, _attr, None)

# ── End of safety preamble ────────────────────────────────────────────────────
'''


# ─────────────────────────────────────────────────────────────────────────────
# Sentinel
# ─────────────────────────────────────────────────────────────────────────────
# A unique token printed to stdout AFTER the test code completes successfully.
# The executor requires this token in stdout (in addition to returncode == 0)
# to report passed=True.
#
# Why this is necessary:
#   returncode == 0 alone is not sufficient — model code could exit cleanly
#   before the tests run (via raise SystemExit(0), which our preamble cannot
#   block because it is a direct exception raise, not a function call).
#   The sentinel is only printed if execution reaches the end of the script,
#   meaning both candidate code AND test code ran to completion without error.
#
# The token includes a fixed hex suffix chosen to make accidental collision
# with model output essentially impossible.
_PASS_SENTINEL = "__REASONINGLAB_TESTS_COMPLETE_7f3a9b2e__"


# ─────────────────────────────────────────────────────────────────────────────
# Result type
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ExecutionResult:
    """Structured outcome of a single candidate-code execution attempt.

    frozen=True makes instances immutable so downstream code (policies,
    runner, logger) cannot accidentally modify a result after it is produced.
    """

    # True only when returncode == 0 AND the pass sentinel was found in stdout.
    # Both conditions are required: returncode alone can be spoofed by early
    # exits (raise SystemExit(0)); the sentinel proves tests ran to completion.
    passed: bool

    # True when the subprocess was killed for exceeding the wall-clock timeout.
    # Explicit flag so taxonomy.py can label timeouts without inspecting stderr
    # (stderr is empty when the process is killed by the OS).
    timed_out: bool

    # Raw text written to stdout (debug prints from the script).
    stdout: str

    # Raw text written to stderr (Python tracebacks and error messages).
    # Read by taxonomy.py to classify the failure type.
    stderr: str

    # Wall-clock seconds from subprocess launch to exit or timeout kill.
    # Used by eval/metrics.py for per-attempt latency accounting.
    elapsed_s: float

    # Process exit code; None when the process was killed by timeout before
    # it could produce one.  Useful for low-level debugging.
    exit_code: int | None


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_script(candidate_code: str, test_code: str) -> str:
    """Concatenate preamble + candidate code + test code into one runnable script.

    Execution order:
      1. Safety preamble  – nullifies dangerous functions first.
      2. Candidate code   – defines the solution function(s).
      3. Test code        – calls assertions against those functions.
      4. Sentinel print   – emitted only if steps 2 and 3 completed without error.

    Two blank lines between blocks follow PEP 8 top-level separation and keep
    tracebacks readable (line numbers map to the combined script).
    """
    # The sentinel line is the very last statement.  It is only reached when
    # candidate code and test code both execute without raising an exception,
    # and without any early exit.  execute_candidate() requires this token in
    # stdout before it will report passed=True.
    sentinel_footer = f'\nprint("{_PASS_SENTINEL}")\n'

    return (
        _SAFETY_PREAMBLE
        + "\n\n"
        + candidate_code.rstrip()
        + "\n\n"
        + test_code.lstrip()
        + sentinel_footer
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def execute_candidate(
    candidate_code: str,
    test_code: str,
    timeout_s: float = 8.0,
) -> ExecutionResult:
    """Run candidate_code + test_code together in an isolated subprocess.

    The combined script is written to a temporary .py file (not passed via
    -c) so that:
      - There are no command-line length limits.
      - Tracebacks show real line numbers (easier for taxonomy classification).
      - sys.executable ensures the same Python environment as the runner.

    Args:
        candidate_code: Model-generated Python code defining the solution.
        test_code:      Deterministic test suite from TaskRecord.test_code.
        timeout_s:      Max wall-clock seconds before the process is killed.

    Returns:
        ExecutionResult capturing pass/fail, raw output, timing, and exit code.
    """
    script = _build_script(candidate_code, test_code)

    tmp_path: Path | None = None
    try:
        # Write the combined script to a named temp file.
        # delete=False: the file must remain on disk until the subprocess
        # finishes reading it; we delete it ourselves in the finally block.
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(script)
            tmp_path = Path(tmp.name)

        # Run the script inside a fresh temporary working directory.
        # Any files the script creates land here and are deleted when the
        # context manager exits — regardless of how the subprocess ends.
        with tempfile.TemporaryDirectory() as work_dir:
            start = time.perf_counter()

            try:
                proc = subprocess.run(
                    [sys.executable, str(tmp_path)],
                    capture_output=True,  # collect stdout and stderr
                    text=True,            # decode bytes to str automatically
                    timeout=timeout_s,    # wall-clock kill (covers infinite loops + sleep hangs)
                    cwd=work_dir,         # isolate file I/O inside temp dir
                )
                elapsed = time.perf_counter() - start

                # Sentinel check: returncode == 0 is necessary but not sufficient.
                # The sentinel is only printed when execution reaches the end of
                # the script, meaning both candidate code and test code ran to
                # completion.  Without this check, raise SystemExit(0) before
                # the tests would still produce returncode == 0 (false "passed").
                sentinel_found = _PASS_SENTINEL in proc.stdout

                # Strip the sentinel token from stdout before storing it.
                # It is internal infrastructure noise; downstream code (logger,
                # metrics, qualitative analysis) should see only model output.
                clean_stdout = proc.stdout.replace(_PASS_SENTINEL, "").rstrip("\n")

                return ExecutionResult(
                    passed=proc.returncode == 0 and sentinel_found,
                    timed_out=False,
                    stdout=clean_stdout,
                    stderr=proc.stderr,
                    elapsed_s=elapsed,
                    exit_code=proc.returncode,
                )

            except subprocess.TimeoutExpired as exc:

                # we need elapsed for congruency with ExecutionResult dataclass but also 
                # for the calculation of latency metric for MLRE experiments
                elapsed = time.perf_counter() - start

                # exc.stdout / exc.stderr may be None if nothing was flushed
                # before the timeout; normalise to empty strings for consistency
                # so taxonomy.py never receives None.
                return ExecutionResult(
                    passed=False,
                    timed_out=True,
                    stdout=exc.stdout or "",
                    stderr=exc.stderr or "",
                    elapsed_s=elapsed,
                    exit_code=None,
                )

    finally:
        # Always remove the temp script file, even if an unexpected exception
        # occurs (e.g. PermissionError writing the file, or TemporaryDirectory
        # failing to create).
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()
