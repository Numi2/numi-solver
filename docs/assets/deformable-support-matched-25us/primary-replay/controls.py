#!/usr/bin/env python3
"""Exercise three independent fail-closed checks in the public spatial replay."""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("asset", type=Path)
    parser.add_argument("--published-spatial", type=Path)
    parser.add_argument("--published-l3-geometry", type=Path)
    parser.add_argument("--published-l4-geometry", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    asset = args.asset.resolve()
    spatial = (args.published_spatial or asset.parent / "spatial.json").resolve()
    l3_geom = (args.published_l3_geometry or asset.parent / "l3-primary-geometry.json").resolve()
    l4_geom = (args.published_l4_geometry or asset.parent.parent / "deformable-support-81920-geometry.json").resolve()
    original = json.loads((asset / "compression-receipt.json").read_text())
    replay = Path(__file__).resolve().with_name("replay.py")
    results = []
    for case in ("energy_byte_change", "truncated_trace_gzip", "false_published_maximum"):
        with tempfile.TemporaryDirectory(prefix="numi-spatial-negative-") as temp:
            folder = Path(temp)
            for name in original["inputs"]:
                (folder / name).symlink_to(asset / name)
            shutil.copy2(asset / "compression-receipt.json", folder / "compression-receipt.json")
            candidate_spatial = spatial
            if case == "energy_byte_change":
                (folder / "l3-energy.csv").unlink()
                shutil.copy2(asset / "l3-energy.csv", folder / "l3-energy.csv")
                with (folder / "l3-energy.csv").open("ab") as f:
                    f.write(b"\n")
            elif case == "truncated_trace_gzip":
                (folder / "l3-primary.csv.gz").unlink()
                (folder / "l3-primary.csv.gz").write_bytes((asset / "l3-primary.csv.gz").read_bytes()[:128])
            else:
                copied = json.loads(spatial.read_text())
                copied["comparison"][0]["maximum_matched_node_difference_m"] = 0.0
                candidate_spatial = folder / "spatial.json"
                candidate_spatial.write_text(json.dumps(copied, indent=2) + "\n")
            command = [sys.executable, str(replay), str(folder),
                       "--published-spatial", str(candidate_spatial),
                       "--published-l3-geometry", str(l3_geom),
                       "--published-l4-geometry", str(l4_geom),
                       "--output", str(folder / "unexpected-success.json")]
            process = subprocess.run(command, capture_output=True, text=True)
            if process.returncode == 0 or (folder / "unexpected-success.json").exists():
                raise RuntimeError(f"negative control was accepted: {case}")
            results.append({"case": case, "actual_exit_code": process.returncode,
                            "failure": process.stderr.strip().splitlines()[-1]})
    result = {"schema": "numi.deformable.matched-L3-L4-25us.replay-negative-controls.v1",
              "replay_sha256": sha(replay),
              "compression_receipt_sha256": sha(asset / "compression-receipt.json"),
              "all_three_rejected": True, "controls": results}
    payload = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(payload)
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
