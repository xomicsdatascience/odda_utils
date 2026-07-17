# Unit tests for odda_utils.sandbox, the least-privilege Apptainer sandbox for
# agent-synthesized analysis code. Exercises the pure helpers (hardened argv
# construction, deterministic code hashing, image/version resolution) and the
# review-gated orchestrator's control flow (preview vs execute, approval-hash
# mismatch, sensitive-path refusal, tamper-evidence). These tests do NOT require
# Apptainer to be installed: every path exercised stops before (or is refused
# before) an actual container launch. Runnable with `python -m unittest` or
# pytest; depends only on the standard library.

import asyncio
import os
import tempfile
import unittest
from pathlib import Path

from odda_utils import sandbox


def _run(coro):
    return asyncio.run(coro)


class TestArgvBuilder(unittest.TestCase):
    def test_hardening_flags_present(self):
        argv = sandbox.build_apptainer_argv(
            sif="/imgs/analysis_v1.0.0.sif",
            work_dir="/runs/r1",
            script_rel="analysis_scratch/de.py",
            dataset_binds=[("PXD1", "/data/PXD1")],
            cpu_seconds=600,
            memory_mb=2048,
            max_file_mb=512,
            allow_network=False,
        )
        self.assertEqual(argv[:2], ["apptainer", "exec"])
        for flag in ("--containall", "--no-home", "--net", "none", "--pwd"):
            self.assertIn(flag, argv)
        # network none must appear as an adjacent --network none pair
        i = argv.index("--network")
        self.assertEqual(argv[i + 1], "none")
        self.assertIn("/runs/r1:/work", argv)
        self.assertIn("/data/PXD1:/data/in/PXD1:ro", argv)

    def test_ulimits_and_entrypoint(self):
        argv = sandbox.build_apptainer_argv(
            sif="s", work_dir="/w", script_rel="a.py",
            cpu_seconds=600, memory_mb=2048, max_file_mb=512,
        )
        inner = argv[-1]
        self.assertEqual(argv[-3:-1], ["/bin/bash", "-c"])
        # bash ulimit units are KiB for -v and -f
        self.assertIn("ulimit -t 600 -v 2097152 -f 524288", inner)
        self.assertTrue(inner.strip().endswith("exec python3 a.py"), inner)

    def test_allow_network_omits_net(self):
        argv = sandbox.build_apptainer_argv(
            sif="s", work_dir="/w", script_rel="a.py", allow_network=True,
        )
        self.assertNotIn("--net", argv)
        self.assertNotIn("--network", argv)

    def test_no_limits_omits_ulimit(self):
        argv = sandbox.build_apptainer_argv(
            sif="s", work_dir="/w", script_rel="a.py",
            cpu_seconds=None, memory_mb=None, max_file_mb=None,
        )
        self.assertNotIn("ulimit", argv[-1])

    def test_paths_are_shell_quoted(self):
        argv = sandbox.build_apptainer_argv(
            sif="s", work_dir="/w", script_rel="sub dir/a.py",
        )
        self.assertIn("'sub dir/a.py'", argv[-1])


class TestCodeHash(unittest.TestCase):
    def test_deterministic_and_tamper_evident(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "pkg").mkdir()
            (root / "a.py").write_text("print(1)\n")
            (root / "pkg" / "b.py").write_text("print(2)\n")
            (root / "data.csv").write_text("x,y\n1,2\n")  # non-.py ignored
            h1, files1 = sandbox.compute_code_hash(root)
            h2, files2 = sandbox.compute_code_hash(root)
            self.assertEqual(h1, h2)
            self.assertEqual(len(h1), 64)
            self.assertEqual(files1, ["a.py", "pkg/b.py"])  # sorted, posix, code-only
            (root / "a.py").write_text("print(1)  # edited\n")
            h3, _ = sandbox.compute_code_hash(root)
            self.assertNotEqual(h1, h3)


class TestVersionResolution(unittest.TestCase):
    def test_env_override_missing_file(self):
        os.environ["ODDA_ANALYSIS_SIF"] = "/nonexistent/analysis.sif"
        try:
            r = sandbox.resolve_analysis_sif()
            self.assertFalse(r["ok"])
            self.assertIn("missing file", r["error"])
        finally:
            del os.environ["ODDA_ANALYSIS_SIF"]

    def test_dir_override_lists_versions(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "analysis_v1.0.0.sif").write_bytes(b"x")
            Path(d, "analysis_v1.2.0.sif").write_bytes(b"x")
            Path(d, "analysis.sif").write_bytes(b"x")
            os.environ["ODDA_ANALYSIS_SIF_DIR"] = d
            try:
                listing = sandbox.list_analysis_versions()
                self.assertTrue(listing["ok"])
                # numeric-aware, newest first; unversioned sorted last
                self.assertEqual(listing["versions"][0], "1.2.0")
                self.assertIn("unversioned", listing["versions"])
                res = sandbox.resolve_analysis_sif()  # newest available
                self.assertTrue(res["ok"])
                self.assertEqual(res["version"], "1.2.0")
                res2 = sandbox.resolve_analysis_sif(version="1.0.0")
                self.assertTrue(res2["ok"])
                self.assertTrue(res2["sif"].endswith("analysis_v1.0.0.sif"))
                res3 = sandbox.resolve_analysis_sif(version="9.9.9")
                self.assertFalse(res3["ok"])
            finally:
                del os.environ["ODDA_ANALYSIS_SIF_DIR"]


class TestOrchestratorGate(unittest.TestCase):
    def _make_run(self, d):
        os.makedirs(os.path.join(d, "analysis_scratch"))
        sp = os.path.join(d, "analysis_scratch", "de.py")
        with open(sp, "w") as f:
            f.write("print('hello')\n")
        return sp

    def test_preview_does_not_execute(self):
        with tempfile.TemporaryDirectory() as d:
            self._make_run(d)
            r = _run(sandbox.run_analysis_sandboxed(d, "analysis_scratch/de.py"))
            self.assertTrue(r["ok"])
            self.assertEqual(r["mode"], "preview")
            self.assertEqual(len(r["code_sha256"]), 64)
            self.assertTrue(r["network_isolated"])
            self.assertIn("de.py", " ".join(r["code_files"]))
            self.assertNotIn("exit_code", r)  # never executed

    def test_approval_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            self._make_run(d)
            r = _run(sandbox.run_analysis_sandboxed(
                d, "analysis_scratch/de.py", approved_code_sha256="deadbeef"))
            self.assertFalse(r["ok"])
            self.assertEqual(r["mode"], "rejected")

    def test_sensitive_path_refused(self):
        with tempfile.TemporaryDirectory() as d:
            cred = os.path.join(d, ".claude")
            os.makedirs(cred)
            r = _run(sandbox.run_analysis_sandboxed(cred, "x.py"))
            self.assertFalse(r["ok"])
            self.assertEqual(r["mode"], "rejected")

    def test_sensitive_dataset_bind_refused(self):
        with tempfile.TemporaryDirectory() as d:
            self._make_run(d)
            key = os.path.join(d, "azure.key")
            with open(key, "w") as f:
                f.write("SECRET\n")
            r = _run(sandbox.run_analysis_sandboxed(
                d, "analysis_scratch/de.py", dataset_paths=[key]))
            self.assertFalse(r["ok"])
            self.assertEqual(r["mode"], "rejected")

    def test_relative_work_dir_refused(self):
        r = _run(sandbox.run_analysis_sandboxed("relative/dir", "x.py"))
        self.assertFalse(r["ok"])
        self.assertEqual(r["mode"], "rejected")

    def test_missing_script_refused(self):
        with tempfile.TemporaryDirectory() as d:
            r = _run(sandbox.run_analysis_sandboxed(d, "does_not_exist.py"))
            self.assertFalse(r["ok"])
            self.assertEqual(r["mode"], "rejected")

    def test_script_escape_refused(self):
        with tempfile.TemporaryDirectory() as d:
            self._make_run(d)
            r = _run(sandbox.run_analysis_sandboxed(d, "../escape.py"))
            self.assertFalse(r["ok"])
            self.assertEqual(r["mode"], "rejected")


if __name__ == "__main__":
    unittest.main()
