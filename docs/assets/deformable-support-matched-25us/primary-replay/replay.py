#!/usr/bin/env python3
"""Independently replay the published matched L3/L4 25 us spatial comparison.

This is a streaming comparison of the two native primary CSV exports. It checks
the owning byte hashes, topology-derived masses, sampled node states, and energy
ledger step grids. It does not replay native integration or audit OBJ surfaces.
"""

import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path


NODES = (2057, 14993)
TETS = (10240, 81920)
TRACE_HEADER = ["frame", "time_s", "node", "x_m", "y_m", "z_m", "vx_m_s", "vy_m_s", "vz_m_s", "mass_kg"]
TOPOLOGY_HEADER = ["element", "node0", "node1", "node2", "node3", "rest_volume_m3", "mu_Pa", "lambda_Pa"]


def hashed(path, compressed):
    compressed_hash = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            compressed_hash.update(chunk)
    raw_hash = hashlib.sha256()
    length = 0
    with (gzip.open(path, "rb") if compressed else path.open("rb")) as f:
        while chunk := f.read(1024 * 1024):
            raw_hash.update(chunk)
            length += len(chunk)
    return compressed_hash.hexdigest(), raw_hash.hexdigest(), length


def verify_inputs(folder, receipt):
    observed = {}
    for name, entry in receipt["inputs"].items():
        archive_sha, raw_sha, size = hashed(folder / name, name.endswith(".gz"))
        if (archive_sha, raw_sha, size) != (entry["archive_sha256"], entry["decompressed_sha256"], entry["uncompressed_bytes"]):
            raise ValueError(f"archive, decompressed hash, or byte count differs: {name}")
        observed[name] = {"archive_sha256": archive_sha, "decompressed_sha256": raw_sha, "uncompressed_bytes": size}
    return observed


def topology(path, node_count, tet_count):
    masses = [0.0] * node_count
    used = set()
    unique = set()
    with gzip.open(path, "rt", newline="") as f:
        reader = csv.reader(f)
        if next(reader) != TOPOLOGY_HEADER:
            raise ValueError(f"unexpected topology header: {path.name}")
        for element, row in enumerate(reader):
            if len(row) != 8 or int(row[0]) != element:
                raise ValueError("topology element sequence or width differs")
            nodes = tuple(int(v) for v in row[1:5])
            if len(set(nodes)) != 4 or any(n < 0 or n >= node_count for n in nodes):
                raise ValueError("invalid tetrahedron corners")
            canonical = tuple(sorted(nodes))
            if canonical in unique:
                raise ValueError("duplicate tetrahedron")
            unique.add(canonical)
            used.update(nodes)
            volume, mu, lam = map(float, row[5:8])
            if not math.isfinite(volume) or volume <= 0 or (mu, lam) != (3000.0, 6000.0):
                raise ValueError("changed rest volume or authored material")
            for n in nodes:
                masses[n] += 1000.0 * volume / 4.0
    if len(unique) != tet_count or used != set(range(node_count)):
        raise ValueError("topology is incomplete")
    return masses


def energy(path):
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        required = {"frame", "time_s", "accepted_steps", "rejected_steps", "integrator", "normal_kinetic_loss_J"}
        if not fields or not required.issubset(fields):
            raise ValueError("missing owning energy fields")
        impact = None
        final_loss = None
        for frame, row in enumerate(reader):
            if len(row) != len(fields) or any(value is None for value in row.values()):
                raise ValueError("malformed energy row")
            values = [float(row[field]) for field in fields]
            if any(not math.isfinite(v) for v in values):
                raise ValueError("nonfinite energy row")
            if (int(row["frame"]), int(row["accepted_steps"]), int(row["rejected_steps"]), int(row["integrator"])) != (frame, 200 * frame, 0, 1):
                raise ValueError("energy ledger changes the accepted support-Verlet grid")
            if abs(float(row["time_s"]) - .005 * frame) > 1e-12:
                raise ValueError("energy frame timestamp differs")
            loss = float(row["normal_kinetic_loss_J"])
            if loss < 0:
                raise ValueError("negative normal kinetic loss")
            if impact is None and loss > 0:
                impact = frame
            final_loss = loss
    if frame != 100:
        raise ValueError("energy ledger lacks 101 complete states")
    return impact, final_loss


def frame_rows(reader, frame, node_count, expected_masses):
    positions = []
    masses = []
    for node in range(node_count):
        row = next(reader, None)
        if row is None or len(row) != 10:
            raise ValueError("truncated or malformed primary trajectory")
        if (int(row[0]), int(row[2])) != (frame, node):
            raise ValueError("nonconsecutive trajectory frame or node")
        values = [float(v) for v in row[1:]]
        if any(not math.isfinite(v) for v in values):
            raise ValueError("nonfinite trajectory state")
        if abs(float(row[1]) - frame * .005) > 1e-12:
            raise ValueError("trajectory timestamp differs")
        mass = float(row[9])
        if abs(mass - expected_masses[node]) > 5e-8:
            raise ValueError("trajectory mass disagrees with volume-derived mass")
        if frame == 0 and any(float(v) != 0.0 for v in row[6:9]):
            raise ValueError("initially stationary source changed")
        positions.append(tuple(float(v) for v in row[3:6]))
        masses.append(mass)
    return positions, masses


def center(positions, masses, total_mass):
    return tuple(legacy_sum(p[k] * m for p, m in zip(positions, masses)) / total_mass for k in range(3))


def legacy_sum(values):
    """Use the source Python 3.9 left-to-right sum on Python 3.12+ too."""
    total = 0.0
    for value in values:
        total += value
    return total


def height(positions):
    return max(p[2] for p in positions) - min(p[2] for p in positions)


def compare(folder, receipt, published):
    coarse_masses = topology(folder / "l3-topology.csv.gz", NODES[0], TETS[0])
    fine_masses = topology(folder / "l4-topology.csv.gz", NODES[1], TETS[1])
    coarse_impact, coarse_loss = energy(folder / "l3-energy.csv")
    fine_impact, fine_loss = energy(folder / "l4-energy.csv")
    impact = min(frame for frame in (coarse_impact, fine_impact) if frame is not None)
    maximum = (-1.0, -1, -1)
    maximum_preimpact = 0.0
    maximum_center = 0.0
    maximum_height = 0.0
    mass_difference = None
    with gzip.open(folder / "l3-primary.csv.gz", "rt", newline="") as a, gzip.open(folder / "l4-primary.csv.gz", "rt", newline="") as b:
        ca, cb = csv.reader(a), csv.reader(b)
        if next(ca) != TRACE_HEADER or next(cb) != TRACE_HEADER:
            raise ValueError("unexpected primary trajectory header")
        for frame in range(101):
            coarse, cm = frame_rows(ca, frame, NODES[0], coarse_masses)
            fine, fm = frame_rows(cb, frame, NODES[1], fine_masses)
            if frame == 0:
                if coarse != fine[:NODES[0]]:
                    raise ValueError("refinement changed pre-existing rest nodes")
                first_cm, first_fm = cm, fm
                total_cm, total_fm = legacy_sum(cm), legacy_sum(fm)
                mass_difference = abs(total_cm - total_fm)
                if mass_difference > 2e-7:
                    raise ValueError("refinement changed total body mass")
            elif cm != first_cm or fm != first_fm:
                raise ValueError("trajectory changed nodal masses")
            frame_error, node = max((math.dist(p, q), n) for n, (p, q) in enumerate(zip(coarse, fine[:NODES[0]])))
            maximum = max(maximum, (frame_error, frame, node))
            if frame < impact:
                maximum_preimpact = max(maximum_preimpact, frame_error)
            maximum_center = max(maximum_center, math.dist(center(coarse, cm, total_cm), center(fine, fm, total_fm)))
            maximum_height = max(maximum_height, abs(height(coarse) - height(fine)))
        if next(ca, None) is not None or next(cb, None) is not None:
            raise ValueError("trajectory contains unexpected extra states")
    node_error, maximum_frame, maximum_node = maximum
    report = {
        "coarse_nodes": NODES[0], "fine_nodes": NODES[1],
        "coarse_tetrahedra": TETS[0], "fine_tetrahedra": TETS[1],
        "total_mass_difference_kg": mass_difference,
        "maximum_matched_node_difference_m": node_error,
        "maximum_difference_frame": maximum_frame,
        "maximum_difference_time_s": maximum_frame * .005,
        "maximum_difference_node": maximum_node,
        "maximum_pre_captured_impact_node_difference_m": maximum_preimpact,
        "coarse_first_captured_impact_frame": coarse_impact,
        "fine_first_captured_impact_frame": fine_impact,
        "coarse_normal_kinetic_loss_J": coarse_loss,
        "fine_normal_kinetic_loss_J": fine_loss,
        "integrator": "support-verlet", "accepted_steps": 20000,
        "maximum_COM_difference_m": maximum_center,
        "maximum_height_difference_m": maximum_height,
        "within_spatial_target": node_error <= published["spatial_target_m"],
        "coarse_trace_sha256": receipt["inputs"]["l3-primary.csv.gz"]["decompressed_sha256"],
        "fine_trace_sha256": receipt["inputs"]["l4-primary.csv.gz"]["decompressed_sha256"],
        "coarse_energy_ledger_sha256": receipt["inputs"]["l3-energy.csv"]["decompressed_sha256"],
        "fine_energy_ledger_sha256": receipt["inputs"]["l4-energy.csv"]["decompressed_sha256"],
    }
    differences = {key: {"measured": value, "published": published["comparison"][0][key]}
                   for key, value in report.items()
                   if value != published["comparison"][0][key] and not (
                       key == "maximum_COM_difference_m" and
                       abs(value - published["comparison"][0][key]) <= 1e-15)}
    if differences:
        raise ValueError(f"independent streaming result differs from published spatial report: {differences}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("asset", type=Path, help="directory containing compressed inputs and compression receipt")
    parser.add_argument("--published-spatial", type=Path, help="published spatial.json; defaults to the parent asset directory")
    parser.add_argument("--published-l3-geometry", type=Path, help="published L3 primary geometry audit; defaults to the parent asset directory")
    parser.add_argument("--published-l4-geometry", type=Path, help="published L4 primary geometry audit; defaults to the sibling asset")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    folder = args.asset
    published_path = args.published_spatial or folder.parent / "spatial.json"
    l3_geometry_path = args.published_l3_geometry or folder.parent / "l3-primary-geometry.json"
    l4_geometry_path = args.published_l4_geometry or folder.parent.parent / "deformable-support-81920-geometry.json"
    receipt = json.loads((folder / "compression-receipt.json").read_text())
    published = json.loads(published_path.read_text())
    geometries = [json.loads(path.read_text()) for path in (l3_geometry_path, l4_geometry_path)]
    if receipt.get("schema") != "numi.deformable.matched-L3-L4-25us.lossless-primary-inputs.v1":
        raise ValueError("unexpected compression receipt schema")
    expected_names = {"l3-primary.csv.gz", "l4-primary.csv.gz", "l3-topology.csv.gz", "l4-topology.csv.gz", "l3-energy.csv", "l4-energy.csv"}
    if set(receipt["inputs"]) != expected_names:
        raise ValueError("compression receipt does not list the exact six primary inputs")
    observed = verify_inputs(folder, receipt)
    if (observed["l3-primary.csv.gz"]["decompressed_sha256"], observed["l4-primary.csv.gz"]["decompressed_sha256"]) != (
            published["comparison"][0]["coarse_trace_sha256"], published["comparison"][0]["fine_trace_sha256"]):
        raise ValueError("raw primary traces disagree with frozen spatial report")
    for level, geometry in zip(("l3", "l4"), geometries):
        if (geometry["nodes"], geometry["tetrahedra"]) != (NODES[0 if level == "l3" else 1], TETS[0 if level == "l3" else 1]):
            raise ValueError(f"{level} published geometry dimensions differ")
        if (observed[f"{level}-primary.csv.gz"]["decompressed_sha256"],
                observed[f"{level}-topology.csv.gz"]["decompressed_sha256"]) != (
                geometry["trace_sha256"], geometry["topology_sha256"]):
            raise ValueError(f"{level} input hashes disagree with published geometry audit")
    report = compare(folder, receipt, published)
    result = {"schema": "numi.deformable.matched-L3-L4-25us.streaming-public-replay.v1",
              "published_spatial_sha256": hashlib.sha256(published_path.read_bytes()).hexdigest(),
              "published_l3_geometry_sha256": hashlib.sha256(l3_geometry_path.read_bytes()).hexdigest(),
              "published_l4_geometry_sha256": hashlib.sha256(l4_geometry_path.read_bytes()).hexdigest(),
              "compression_receipt_sha256": hashlib.sha256((folder / "compression-receipt.json").read_bytes()).hexdigest(),
              "inputs_verified": observed, "comparison": report,
              "matches_published_spatial_result": True,
              "evidence_boundary": "Exact lossless native primary CSV and topology inputs, owning sampled energy ledgers, and independently recomputed sampled spatial metrics. No native re-integration, OBJ surface re-audit, material calibration, fruit coupling, or continuous-time convergence proof."}
    payload = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(payload)
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
