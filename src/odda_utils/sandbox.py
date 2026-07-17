# Least-privilege Apptainer sandbox for executing agent-synthesized analysis code.
#
# This module implements the "synthesis sandbox" specified in the repository
# threat model (SECURITY_THREAT_MODEL.md, section 5): the one ODDA stage that
# runs code derived from untrusted article text. Code produced at cross-study
# synthesis / downstream analysis is treated as untrusted until a human has read
# it, so execution here is (a) gated behind a tamper-evident review hash and
# (b) confined to a hardened Apptainer container:
#
#   * --containall + --no-home : no host filesystems, clean env, isolated
#     PID/IPC namespaces, and crucially NO $HOME mount (keeps credentials in
#     ~/.claude out of the container).
#   * --net --network none     : network egress disabled by default, which
#     neutralizes the exfiltration and download-and-run categories the
#     injection telemetry flags. If the platform cannot create an isolated
#     network namespace unprivileged, the run FAILS CLOSED rather than running
#     with host networking (override only with allow_network=True).
#   * read-only root filesystem: the SIF image is immutable; the only writable
#     path is a single scratch bind (the run's working directory at /work).
#   * least-privilege data     : only the datasets explicitly named are bind
#     mounted, read-only, under /data/in/; the database and credential files are
#     never mounted.
#   * resource limits          : CPU-time, address-space (memory), and file-size
#     caps via `ulimit` (robust on hosts without working cgroups), a host-side
#     wall-clock timeout that hard-kills the process, and a cap on captured
#     output bytes.
#
# The honest position stated in the paper is that the secure way to run
# possibly-malicious code is not to run it unreviewed; this sandbox bounds the
# damage if review is imperfect. Pure helpers (argv construction, code hashing,
# version resolution) are separated from I/O so they can be unit-tested without
# Apptainer installed.

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import os
import re
import shlex
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# --- Defaults (all overridable per call) -----------------------------------
DEFAULT_WALL_CLOCK_SEC: int = 3600
DEFAULT_CPU_SECONDS: Optional[int] = None
DEFAULT_MEMORY_MB: Optional[int] = 4096
DEFAULT_MAX_FILE_MB: Optional[int] = 2048
DEFAULT_MAX_OUTPUT_BYTES: int = 1_000_000

# In-container mount points.
WORK_MOUNT = "/work"
DATA_MOUNT = "/data/in"

_SIF_NAME_RE = re.compile(r"^analysis_v(.+)\.sif$")
# Substrings that must never appear in a bind-mounted host path: credentials,
# the SQLite database, and Claude Code config all live under these.
_FORBIDDEN_BIND_SUBSTRINGS = (".claude",)
_FORBIDDEN_BIND_SUFFIXES = (".sqlite", ".sqlite-journal", ".key", ".endpoint")


# ---------------------------------------------------------------------------
# Image resolution (mirrors odda_salmon's .sif discovery)
# ---------------------------------------------------------------------------
def _analysis_sif_dir() -> Path:
    """Return the directory that holds the built analysis Apptainer image(s).

    Defaults to the package-relative ``static/apptainer`` directory. Overridable
    with the ``ODDA_ANALYSIS_SIF_DIR`` environment variable when images are
    stored outside the source tree.

    Returns
    -------
    pathlib.Path
        Directory expected to contain ``analysis_v*.sif`` (or ``analysis.sif``).
    """
    override = os.environ.get("ODDA_ANALYSIS_SIF_DIR")
    if override:
        return Path(override)
    # sandbox.py lives at odda_utils/src/odda_utils/sandbox.py, so the package
    # root (odda_utils) is three parents up.
    return Path(__file__).resolve().parents[2] / "static" / "apptainer"


def _version_key(version: str) -> Tuple:
    """Sortable key for a version string (numeric components compared as ints)."""
    parts = re.split(r"[._-]", version)
    key: List[Tuple[int, Any]] = []
    for p in parts:
        if p.isdigit():
            key.append((0, int(p)))
        else:
            key.append((1, p))
    return tuple(key)


def list_analysis_versions() -> Dict[str, Any]:
    """List analysis-container versions discoverable from built images.

    Returns
    -------
    dict
        ``{"ok": True, "versions": [...], "sif_dir": <str>}`` on success, sorted
        newest-first. ``versions`` may include ``"unversioned"`` if a plain
        ``analysis.sif`` is present.
    """
    sif_dir = _analysis_sif_dir()
    versions: List[str] = []
    unversioned = False
    if sif_dir.is_dir():
        for p in sif_dir.iterdir():
            if not p.is_file():
                continue
            m = _SIF_NAME_RE.match(p.name)
            if m:
                versions.append(m.group(1))
            elif p.name == "analysis.sif":
                unversioned = True
    versions.sort(key=_version_key, reverse=True)
    if unversioned:
        versions.append("unversioned")
    return {"ok": True, "versions": versions, "sif_dir": str(sif_dir)}


def resolve_analysis_sif(version: Optional[str] = None) -> Dict[str, Any]:
    """Resolve a concrete analysis ``.sif`` image path.

    Resolution order: the ``ODDA_ANALYSIS_SIF`` environment variable (a direct
    path), then ``analysis_v{version}.sif`` for an explicit ``version``, then the
    newest ``analysis_v*.sif`` in the image directory, then a plain
    ``analysis.sif``.

    Parameters
    ----------
    version : str, optional
        Bare image version (e.g. ``"1.0.0"``). If omitted, the newest available
        image is auto-selected.

    Returns
    -------
    dict
        ``{"ok": True, "sif": <str>, "version": <str>}`` on success, or
        ``{"ok": False, "error": <str>}`` if no matching image is found.
    """
    direct = os.environ.get("ODDA_ANALYSIS_SIF")
    if direct:
        p = Path(direct)
        if p.is_file():
            m = _SIF_NAME_RE.match(p.name)
            return {"ok": True, "sif": str(p), "version": m.group(1) if m else "unversioned"}
        return {"ok": False, "error": f"ODDA_ANALYSIS_SIF points to a missing file: {direct}"}

    sif_dir = _analysis_sif_dir()
    if version:
        cand = sif_dir / f"analysis_v{version}.sif"
        if cand.is_file():
            return {"ok": True, "sif": str(cand), "version": version}
        return {
            "ok": False,
            "error": (
                f"No analysis image found for version {version!r} in {sif_dir}. "
                "Build it with static/apptainer/build_images.sh or call "
                "list_analysis_versions to see what is available."
            ),
        }

    listing = list_analysis_versions()
    for v in listing["versions"]:
        if v == "unversioned":
            cand = sif_dir / "analysis.sif"
        else:
            cand = sif_dir / f"analysis_v{v}.sif"
        if cand.is_file():
            return {"ok": True, "sif": str(cand), "version": v}
    return {
        "ok": False,
        "error": (
            f"No analysis Apptainer image found in {sif_dir}. Build one with "
            "static/apptainer/build_images.sh (produces analysis_v<version>.sif)."
        ),
    }


# ---------------------------------------------------------------------------
# Review-hash gate (tamper-evident "human read the code before it ran")
# ---------------------------------------------------------------------------
def compute_code_hash(code_root: Path) -> Tuple[str, List[str]]:
    """Compute a deterministic SHA-256 over every ``*.py`` file under a directory.

    The hash binds the exact bytes of the analysis code that will execute, so an
    operator who reviews the code can approve it by its hash; if the code is
    altered afterwards the hash changes and execution is refused.

    Parameters
    ----------
    code_root : pathlib.Path
        Directory whose ``*.py`` files constitute the analysis code (typically
        the run's working directory).

    Returns
    -------
    (str, list of str)
        The hex digest and the sorted list of hashed file paths (relative,
        POSIX-style). If no ``*.py`` files exist, the digest is of empty content
        and the list is empty.
    """
    files = sorted(p for p in code_root.rglob("*.py") if p.is_file())
    h = hashlib.sha256()
    rels: List[str] = []
    for f in files:
        rel = f.relative_to(code_root).as_posix()
        rels.append(rel)
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(f.read_bytes())
        h.update(b"\0")
    return h.hexdigest(), rels


# ---------------------------------------------------------------------------
# Command construction (pure; unit-testable without Apptainer)
# ---------------------------------------------------------------------------
def build_apptainer_argv(
    *,
    sif: str,
    work_dir: str,
    script_rel: str,
    dataset_binds: Sequence[Tuple[str, str]] = (),
    cpu_seconds: Optional[int] = DEFAULT_CPU_SECONDS,
    memory_mb: Optional[int] = DEFAULT_MEMORY_MB,
    max_file_mb: Optional[int] = DEFAULT_MAX_FILE_MB,
    allow_network: bool = False,
    python_args: Sequence[str] = (),
) -> List[str]:
    """Build the hardened ``apptainer exec`` argv for one analysis run.

    Parameters
    ----------
    sif : str
        Path to the analysis ``.sif`` image.
    work_dir : str
        Host directory bind-mounted read-write at ``/work`` (the only writable
        path). Holds the analysis code and receives outputs.
    script_rel : str
        Entry script path relative to ``work_dir`` (e.g. ``analysis_scratch/de.py``).
    dataset_binds : sequence of (name, host_path)
        Datasets to bind read-only under ``/data/in/<name>``.
    cpu_seconds : int, optional
        CPU-time cap (``ulimit -t``); omitted if None.
    memory_mb : int, optional
        Address-space cap in MiB (``ulimit -v``); omitted if None/0. Note that
        ``-v`` limits virtual address space, which is conservative for
        numpy/pandas; set None to disable if it interferes.
    max_file_mb : int, optional
        Per-file size cap in MiB (``ulimit -f``); omitted if None/0.
    allow_network : bool
        If False (default) add ``--net --network none`` to disable networking.
    python_args : sequence of str
        Extra arguments passed to the analysis script.

    Returns
    -------
    list of str
        The argv to execute. Uses ``bash -c`` inside the container so that
        ``ulimit`` (KiB units in bash) is applied before the interpreter starts.
    """
    ulimits: List[str] = []
    if cpu_seconds:
        ulimits.append(f"-t {int(cpu_seconds)}")
    if memory_mb:
        ulimits.append(f"-v {int(memory_mb) * 1024}")
    if max_file_mb:
        ulimits.append(f"-f {int(max_file_mb) * 1024}")

    inner = ""
    if ulimits:
        inner += "ulimit " + " ".join(ulimits) + "; "
    inner += f"cd {shlex.quote(WORK_MOUNT)} && exec python3 {shlex.quote(script_rel)}"
    if python_args:
        inner += " " + " ".join(shlex.quote(a) for a in python_args)

    argv: List[str] = [
        "apptainer", "exec",
        "--containall",   # no host binds, clean env, isolated PID/IPC namespaces
        "--no-home",      # never mount $HOME (keeps ~/.claude credentials out)
        "--pwd", WORK_MOUNT,
    ]
    if not allow_network:
        # Create a private network namespace with no interfaces. Fails closed on
        # platforms that cannot do this unprivileged (see run_analysis_sandboxed).
        argv += ["--net", "--network", "none"]
    argv += ["--bind", f"{work_dir}:{WORK_MOUNT}"]
    for name, host in dataset_binds:
        argv += ["--bind", f"{host}:{DATA_MOUNT}/{name}:ro"]
    argv += [sif, "/bin/bash", "-c", inner]
    return argv


# ---------------------------------------------------------------------------
# Path validation (defense in depth)
# ---------------------------------------------------------------------------
def _reject_sensitive(path: Path) -> Optional[str]:
    """Return an error string if ``path`` looks like a credential/db location."""
    s = path.as_posix().lower()
    for frag in _FORBIDDEN_BIND_SUBSTRINGS:
        if frag in s.split("/"):
            return f"refusing to mount a path containing {frag!r}: {path}"
    for suf in _FORBIDDEN_BIND_SUFFIXES:
        if s.endswith(suf):
            return f"refusing to mount a {suf} file: {path}"
    return None


# ---------------------------------------------------------------------------
# Capped, timed subprocess capture
# ---------------------------------------------------------------------------
async def _read_capped(stream: asyncio.StreamReader, cap: int) -> Tuple[bytes, bool]:
    """Read a stream up to ``cap`` bytes; drain the rest. Returns (data, truncated)."""
    buf = bytearray()
    truncated = False
    while True:
        chunk = await stream.read(65536)
        if not chunk:
            break
        if len(buf) < cap:
            take = cap - len(buf)
            buf.extend(chunk[:take])
            if len(chunk) > take:
                truncated = True
        else:
            truncated = True
    return bytes(buf), truncated


async def _run_capped(
    argv: Sequence[str],
    *,
    timeout_sec: Optional[float],
    max_output_bytes: int,
) -> Dict[str, Any]:
    """Run ``argv``, capturing at most ``max_output_bytes`` of each stream.

    Enforces the wall-clock timeout by hard-killing the process group on expiry.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError as e:
        return {
            "exit_code": None,
            "stdout": "",
            "stderr": f"apptainer not found; is it installed and on PATH? {e}",
            "timed_out": False,
            "stdout_truncated": False,
            "stderr_truncated": False,
        }

    async def _gather():
        out = await _read_capped(proc.stdout, max_output_bytes)
        err = await _read_capped(proc.stderr, max_output_bytes)
        rc = await proc.wait()
        return out, err, rc

    try:
        (out_b, out_t), (err_b, err_t), rc = await (
            asyncio.wait_for(_gather(), timeout=timeout_sec) if timeout_sec else _gather()
        )
    except asyncio.TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        with contextlib.suppress(Exception):
            await proc.wait()
        return {
            "exit_code": None,
            "stdout": "",
            "stderr": f"Timed out after {timeout_sec}s (wall-clock limit); process killed.",
            "timed_out": True,
            "stdout_truncated": False,
            "stderr_truncated": False,
        }

    return {
        "exit_code": rc,
        "stdout": out_b.decode("utf-8", "replace"),
        "stderr": err_b.decode("utf-8", "replace"),
        "timed_out": False,
        "stdout_truncated": out_t,
        "stderr_truncated": err_t,
    }


def _looks_like_network_setup_failure(stderr: str) -> bool:
    """Heuristic: did apptainer fail because it could not set up the netns?"""
    s = stderr.lower()
    needles = (
        "network", "netns", "cni", "setuid", "operation not permitted",
        "unable to create", "failed to create namespace",
    )
    return any(n in s for n in needles)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
async def run_analysis_sandboxed(
    work_dir: str,
    script: str,
    *,
    dataset_paths: Optional[Sequence[str]] = None,
    approved_code_sha256: Optional[str] = None,
    cpu_seconds: Optional[int] = DEFAULT_CPU_SECONDS,
    memory_mb: Optional[int] = DEFAULT_MEMORY_MB,
    max_file_mb: Optional[int] = DEFAULT_MAX_FILE_MB,
    wall_clock_sec: Optional[int] = DEFAULT_WALL_CLOCK_SEC,
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
    allow_network: bool = False,
    version: Optional[str] = None,
    scan_code: bool = True,
) -> Dict[str, Any]:
    """Execute agent-synthesized analysis code inside the hardened sandbox.

    Two-phase, review-gated:

    * **Preview** (``approved_code_sha256`` is None): validate inputs, hash the
      code, scan it with the injection telemetry, and return the hash plus the
      exact command that *would* run -- WITHOUT executing. The caller (a human,
      or an agent surfacing to a human) reviews the code and re-invokes with
      ``approved_code_sha256`` set to the returned hash.
    * **Execute** (``approved_code_sha256`` matches the current code hash): run
      the code in the container. A mismatch is refused (the code changed since
      review).

    Parameters
    ----------
    work_dir : str
        Host directory bind-mounted read-write at ``/work``; contains the code
        and receives outputs. Must exist and must not be a credential/db path.
    script : str
        Entry script relative to ``work_dir`` (e.g. ``analysis_scratch/de.py``).
    dataset_paths : sequence of str, optional
        Host dataset files/dirs to bind read-only under ``/data/in/<basename>``.
    approved_code_sha256 : str, optional
        The reviewed code hash. None -> preview only.
    cpu_seconds, memory_mb, max_file_mb : int, optional
        Resource caps (see :func:`build_apptainer_argv`).
    wall_clock_sec : int, optional
        Host-side hard timeout in seconds.
    max_output_bytes : int
        Cap on captured stdout/stderr bytes (each).
    allow_network : bool
        If True, do NOT isolate the network (escape hatch; default False).
    version : str, optional
        Analysis image version; newest available if omitted.
    scan_code : bool
        If True, run ``scan_injection`` over the code and include the signal.

    Returns
    -------
    dict
        Structured result. Always includes ``ok`` and ``mode``
        ("preview" | "executed" | "rejected"). Preview adds ``code_sha256``,
        ``code_files``, ``injection_scan``, and ``planned_command``. Execution
        adds ``exit_code``, ``stdout``, ``stderr``, ``timed_out``, truncation
        flags, ``code_sha256``, and ``sif_version``.
    """
    work = Path(work_dir)
    if not work.is_absolute():
        return {"ok": False, "mode": "rejected", "error": f"work_dir must be an absolute path: {work_dir}"}
    if not work.is_dir():
        return {"ok": False, "mode": "rejected", "error": f"work_dir does not exist: {work_dir}"}
    work = work.resolve()
    err = _reject_sensitive(work)
    if err:
        return {"ok": False, "mode": "rejected", "error": err}

    script_path = (work / script).resolve()
    if not str(script_path).startswith(str(work) + os.sep) and script_path != work:
        return {"ok": False, "mode": "rejected", "error": f"script must live inside work_dir: {script}"}
    if not script_path.is_file():
        return {"ok": False, "mode": "rejected", "error": f"entry script not found: {script}"}
    script_rel = script_path.relative_to(work).as_posix()

    # Resolve dataset binds (read-only), rejecting sensitive locations.
    dataset_binds: List[Tuple[str, str]] = []
    resolved_inputs: List[str] = []
    seen_names: Dict[str, int] = {}
    for raw in dataset_paths or []:
        p = Path(raw)
        if not p.exists():
            return {"ok": False, "mode": "rejected", "error": f"dataset path does not exist: {raw}"}
        p = p.resolve()
        serr = _reject_sensitive(p)
        if serr:
            return {"ok": False, "mode": "rejected", "error": serr}
        name = p.name
        # Disambiguate duplicate basenames.
        if name in seen_names:
            seen_names[name] += 1
            name = f"{name}_{seen_names[name]}"
        else:
            seen_names[name] = 0
        dataset_binds.append((name, str(p)))
        resolved_inputs.append(str(p))

    code_sha256, code_files = compute_code_hash(work)

    scan_result: Optional[Dict[str, Any]] = None
    if scan_code:
        with contextlib.suppress(Exception):
            from odda_utils.injection_scan import scan_injection_batch
            items = {rel: (work / rel).read_text("utf-8", "replace") for rel in code_files}
            batch = scan_injection_batch(items) if items else None
            if batch is not None:
                # Reduce to a compact, JSON-friendly signal.
                scan_result = {
                    "flagged_labels": list(getattr(batch, "flagged_labels", []) or []),
                    "max_risk_level": _max_risk_level(batch),
                }

    # ---- Preview phase -----------------------------------------------------
    resolved = resolve_analysis_sif(version)
    planned = None
    if resolved.get("ok"):
        planned = build_apptainer_argv(
            sif=resolved["sif"],
            work_dir=str(work),
            script_rel=script_rel,
            dataset_binds=dataset_binds,
            cpu_seconds=cpu_seconds,
            memory_mb=memory_mb,
            max_file_mb=max_file_mb,
            allow_network=allow_network,
        )

    if approved_code_sha256 is None:
        out: Dict[str, Any] = {
            "ok": True,
            "mode": "preview",
            "code_sha256": code_sha256,
            "code_files": code_files,
            "injection_scan": scan_result,
            "image": resolved,
            "planned_command": planned,
            "network_isolated": not allow_network,
            "message": (
                "Review the code above, then re-invoke run_analysis with "
                f"approved_code_sha256='{code_sha256}' to execute it in the "
                "sandbox. The hash binds the exact code reviewed; any edit "
                "changes it and requires re-review."
            ),
        }
        return out

    # ---- Approval check ----------------------------------------------------
    if approved_code_sha256 != code_sha256:
        return {
            "ok": False,
            "mode": "rejected",
            "code_sha256": code_sha256,
            "error": (
                "Approval hash does not match the current code hash; the code "
                "changed since it was reviewed. Re-review and approve "
                f"code_sha256='{code_sha256}'."
            ),
        }

    if not resolved.get("ok"):
        return {"ok": False, "mode": "rejected", "code_sha256": code_sha256, **resolved}

    # ---- Execute -----------------------------------------------------------
    result = await _run_capped(
        planned, timeout_sec=wall_clock_sec, max_output_bytes=max_output_bytes
    )

    if (
        not allow_network
        and result.get("exit_code") not in (0, None)
        and _looks_like_network_setup_failure(result.get("stderr", ""))
    ):
        result["stderr"] += (
            "\n\n[odda sandbox] The container failed to start with an isolated "
            "network namespace (--net --network none). Unprivileged network "
            "isolation requires setuid-mode Apptainer or administrator "
            "configuration. FAILING CLOSED rather than running with host "
            "networking. To proceed without network isolation (NOT recommended "
            "for untrusted code), pass allow_network=True explicitly."
        )
        result["network_isolation_failed"] = True

    return {
        "ok": result.get("exit_code") == 0,
        "mode": "executed",
        "code_sha256": code_sha256,
        "sif_version": resolved.get("version"),
        "network_isolated": not allow_network,
        "input_paths": resolved_inputs,
        "output_paths": [str(work)],
        "injection_scan": scan_result,
        "command": planned,
        **result,
    }


def _max_risk_level(batch: Any) -> str:
    """Extract the highest per-item risk_level from an InjectionScanBatchResult."""
    order = {"none": 0, "low": 1, "medium": 2, "high": 3}
    best = "none"
    results = getattr(batch, "results", None) or {}
    for r in results.values():
        lvl = getattr(r, "risk_level", "none")
        if order.get(lvl, 0) > order.get(best, 0):
            best = lvl
    return best
