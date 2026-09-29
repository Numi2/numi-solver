#!/usr/bin/env python3
"""Extract the complete reviewed FEM payload without changing retained inputs."""
import gzip
import hashlib
import json
from pathlib import Path
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def main():
    manifest = json.loads((HERE / "payload-manifest.json").read_text())
    decoded = {}
    for name, item in manifest["payloads"].items():
        compressed = (HERE / item["compressed_file"]).read_bytes()
        if hashlib.sha256(compressed).hexdigest() != item["compressed_sha256"]:
            raise ValueError(f"Changed compressed payload: {name}")
        raw = gzip.decompress(compressed)
        if len(raw) != item["bytes"] or hashlib.sha256(raw).hexdigest() != item["sha256"]:
            raise ValueError(f"Changed represented payload: {name}")
        decoded[name] = raw
    body = json.loads(decoded["body.json"])
    if (len(body["nodes"]), len(body["elements"]), len(body["boundary_faces"]),
            len(body["node_incidence_entries"])) != (2057, 10240, 1280, 40960):
        raise ValueError("Unexpected prepared body layout")
    build = ROOT / "build"
    build.mkdir(exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="full-fem-body-public-", dir=build))
    for name, raw in decoded.items():
        (work / name).write_bytes(raw)
    receipt = {"schema": "numi.elastic-yarn.full-fem-body.public-extraction.v1",
        "payload_manifest_sha256": hashlib.sha256((HERE / "payload-manifest.json").read_bytes()).hexdigest(),
        "unpacker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "payloads_verified": len(decoded), "output_directory": str(work),
        "nodes": 2057, "tetrahedra": 10240, "boundary_faces": 1280,
        "source_revision": body["physics_source_revision"], "recorded_frame": body["recorded_frame"],
        "inverse_rows_are_source_reconstructed": True,
        "historical_native_checkpoint_identity_claimed": False,
        "FEM_step_executed": False, "GPU_execution": False}
    (work / "extraction-evidence.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
