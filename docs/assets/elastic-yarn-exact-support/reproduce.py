#!/usr/bin/env python3
"""Rebuild the published CPU certificate in a fresh evidence directory.

Historical qualification receipts remain unchanged. This runner checks public
source copies, rather than unavailable original binary/capture manifests.
"""
import hashlib
import json
import platform
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("This bridge build is qualified for Apple silicon macOS.")
    manifest = json.loads((HERE / "source-manifest.json").read_text())
    for name, item in manifest["files"].items():
        path = HERE / name
        if digest(path) != item["sha256"] or path.stat().st_size != item["bytes"]:
            raise RuntimeError(f"Changed published input: {name}")
    build = ROOT / "build"
    build.mkdir(exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="exact-support-public-", dir=build))
    for name in manifest["files"]:
        target = work / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(HERE / name, target)
    runs = []

    def run(name, argv, expected):
        result = subprocess.run(argv, cwd=work, capture_output=True, timeout=30)
        (work / f"{name}.log").write_bytes(result.stdout + result.stderr)
        (work / f"{name}.exit").write_text(f"{result.returncode}\n")
        runs.append({"name": name, "argv": argv, "actual_exit": result.returncode,
                     "expected_exit": expected, "log_sha256": digest(work / f"{name}.log")})
        if result.returncode != expected:
            raise RuntimeError(f"{name}: actual exit {result.returncode}, expected {expected}; see {work}")

    run("public-compile", ["clang++", "-std=gnu++2b", "-O3", "-DNDEBUG", "-arch", "arm64",
        "-Wall", "-Wextra", "-Wpedantic", "-fno-fast-math", "-ffp-contract=off",
        "-Ifrozen/include", "-shared", "-fPIC", "oracle_bridge.cpp", "-o", "oracle_bridge.dylib"], 0)
    # The old core is executed only as a rejected classifier. It needs the same
    # bridge in its own module directory; no historical binary identity is claimed.
    shutil.copyfile(work / "oracle_bridge.dylib",
                    work / "history/563f6438-before-binding-repair/oracle_bridge.dylib")
    run("public-qualification", [sys.executable, "-c",
        "import qualify as q; q.snapshot_check = lambda: 0; raise SystemExit(q.run())"], 0)
    summary = json.loads((work / "controls.json").read_text())["summary"]
    historical = json.loads((HERE / "controls.json").read_text())["summary"]
    # Zero explicitly records that historical original/frozen paths were not
    # reverified. All other observations must match the reviewed qualification.
    expected_summary = dict(historical)
    expected_summary["immutable_original_and_frozen_files_verified"] = 0
    if summary != expected_summary:
        raise RuntimeError(f"Public qualification differs from reviewed observations; see {work}")
    for mode in ("nan-plane", "rounded-penetration", "endpoint-rotation", "remaining-frontier", "mutable-binding"):
        run(f"public-negative-{mode}", [sys.executable, "qualify.py", "--negative", mode], 1)
    run("public-independent-controls", [sys.executable, "independent_controls.py"], 0)
    run("public-binding-controls", [sys.executable, "binding_controls.py"], 0)
    for name, item in manifest["files"].items():
        if digest(HERE / name) != item["sha256"]:
            raise RuntimeError(f"Published input changed during reproduction: {name}")
    evidence = {"schema": "numi.elastic-yarn.exact-support.public-rerun.v1",
        "public_source_manifest_sha256": digest(HERE / "source-manifest.json"),
        "runner_sha256": digest(Path(__file__)), "published_inputs_verified": len(manifest["files"]),
        "owned_bridge_sha256": digest(work / "oracle_bridge.dylib"),
        "work_directory": str(work), "python": sys.version, "platform": platform.platform(),
        "runs": runs, "summary": summary,
        "independent_checks": json.loads((work / "independent-controls.json").read_text()),
        "binding_checks": json.loads((work / "binding_controls.json").read_text()),
        "historical_original_binaries_reverified": False,
        "GPU_execution": False, "trajectory_execution": False, "coupled_response_qualified": False}
    (work / "public-rerun-evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps({"evidence": str(work / "public-rerun-evidence.json"),
                      "cases": summary["cases"], "actual_qualification_exit": 0,
                      "negative_runs_with_actual_exit_1": 5, "CPU_only": True}))


if __name__ == "__main__":
    main()
