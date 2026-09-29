#!/usr/bin/env python3
"""Make lossless, hash-bound public inputs for the matched L3/L4 drop."""

import argparse
import gzip
import hashlib
import json
import shutil
from pathlib import Path


INPUTS = {
    "l3-primary.csv": ("mesh4-level3-dt25us-match.csv", "4c6189c28e7df1b0ea8b54e20b73b37675f70669d8015fdedb1f2dbebc6f0523", True),
    "l4-primary.csv": ("mesh4-level4-dt25us.csv", "18cb297d2cbe6bd3cd6305d3b0b3931718c75f01f20ebeeb32c565df04b2e25a", True),
    "l3-topology.csv": ("mesh4-level3-dt25us-match-topology.csv", "528bf55f329ad3b702c39047b16f39e08c6d1f13c58cec4afc2e4acc593aa3f1", True),
    "l4-topology.csv": ("mesh4-level4-dt25us-topology.csv", "76dc32845a503d41d389165fe0cc84ed6762afd2fd53c5b01c1810889075755c", True),
    "l3-energy.csv": ("mesh4-level3-dt25us-match-energy.csv", "5699c203f91dcbcd1fca779a6e6800610e4eda705c728698c0eab4d5b27ece07", False),
    "l4-energy.csv": ("mesh4-level4-dt25us-energy.csv", "511cdf4844627ca8535eff392c468ad0af91fa5dcc549e3a6cdbdfd59537b384", False),
}


def digest(path, opener=open):
    h = hashlib.sha256()
    size = 0
    with opener(path, "rb") as f:
        while block := f.read(1024 * 1024):
            h.update(block)
            size += len(block)
    return h.hexdigest(), size


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_build", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    receipt = {"schema": "numi.deformable.matched-L3-L4-25us.lossless-primary-inputs.v1", "inputs": {}}
    for target, (source, expected, compressed) in INPUTS.items():
        source_path = args.source_build / source
        before, source_size = digest(source_path)
        if before != expected:
            raise ValueError(f"source hash changed for {source}: {before}")
        target_path = args.output / (target + (".gz" if compressed else ""))
        with source_path.open("rb") as src, target_path.open("wb") as raw:
            if compressed:
                with gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=6, mtime=0) as gz:
                    shutil.copyfileobj(src, gz, length=1024 * 1024)
            else:
                shutil.copyfileobj(src, raw, length=1024 * 1024)
        after, _ = digest(source_path)
        restored, restored_size = digest(target_path, gzip.open if compressed else open)
        if after != expected or restored != expected or restored_size != source_size:
            raise ValueError(f"lossless identity failed for {source}")
        archive_sha, archive_size = digest(target_path)
        receipt["inputs"][target_path.name] = {
            "owning_source": "build/" + source,
            "source_sha256_before": before,
            "source_sha256_after": after,
            "decompressed_sha256": restored,
            "uncompressed_bytes": source_size,
            "archive_sha256": archive_sha,
            "archive_bytes": archive_size,
        }
    (args.output / "compression-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({name: row["archive_bytes"] for name, row in receipt["inputs"].items()}, indent=2))


if __name__ == "__main__":
    main()
