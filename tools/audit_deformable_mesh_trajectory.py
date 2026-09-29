#!/usr/bin/env python3
"""Audit native shared-node mesh-drop geometry, incidence, and lumped mass."""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path

FIELDS = ['frame', 'time_s', 'node', 'x_m', 'y_m', 'z_m',
          'vx_m_s', 'vy_m_s', 'vz_m_s', 'mass_kg']


def volume(points):
    a, b, c = [tuple(points[n][k] - points[0][k] for k in range(3))
               for n in (1, 2, 3)]
    return (a[0] * (b[1] * c[2] - b[2] * c[1]) +
            a[1] * (b[2] * c[0] - b[0] * c[2]) +
            a[2] * (b[0] * c[1] - b[1] * c[0])) / 6


def audit(prefix):
    trace = Path(str(prefix) + '.csv').read_bytes()
    reader = csv.DictReader(io.StringIO(trace.decode()))
    if reader.fieldnames != FIELDS:
        raise ValueError('unexpected trajectory schema')
    rows = list(reader)
    node_count = sum(int(row['frame']) == 0 for row in rows)
    contracts = {13: (20, 20), 55: (160, 80), 309: (1280, 320), 2057: (10240, 1280)}
    if node_count not in contracts or len(rows) != 101 * node_count:
        raise ValueError('expected 101 complete supported mesh states')
    element_count, boundary_count = contracts[node_count]
    topology = Path(str(prefix) + '-topology.csv').read_bytes()
    tets = list(csv.DictReader(io.StringIO(topology.decode())))
    if len(tets) != element_count:
        raise ValueError('tetrahedron count does not match refinement topology')
    corners, volumes, masses = [], [], [0.] * node_count
    face_owners = {}
    for index, tet in enumerate(tets):
        nodes = tuple(int(tet[f'node{n}']) for n in range(4))
        if int(tet['element']) != index or len(set(nodes)) != 4 or any(n < 0 or n >= node_count for n in nodes):
            raise ValueError('invalid or duplicated tetrahedron corners')
        v = float(tet['rest_volume_m3'])
        if not math.isfinite(v) or v <= 0 or float(tet['mu_Pa']) != 3000 or float(tet['lambda_Pa']) != 6000:
            raise ValueError('invalid rest volume or changed authored material')
        volumes.append(v)
        corners.append(nodes)
        a, b, c, d = (n + 1 for n in nodes)
        for face in ((b, c, d), (a, d, c), (a, b, d), (a, c, b)):
            face_owners.setdefault(tuple(sorted(face)), []).append(face)
        for n in nodes:
            masses[n] += 1000 * v / 4
    if len({tuple(sorted(tet)) for tet in corners}) != element_count:
        raise ValueError('duplicate tetrahedron')
    if set(n for tet in corners for n in tet) != set(range(node_count)):
        raise ValueError('isolated or missing node')
    canonical = lambda face: min(face, face[1:] + face[:1], face[2:] + face[:2])
    faces = []
    for owners in face_owners.values():
        if len(owners) == 1:
            faces.append(owners[0])
        elif len(owners) != 2 or canonical(owners[0]) != canonical(tuple(reversed(owners[1]))):
            raise ValueError('internal face lacks exactly two opposing element owners')
    if len(faces) != boundary_count:
        raise ValueError('element boundary count mismatch')
    edges = {}
    for face in faces:
        for a, b in zip(face, face[1:] + face[:1]):
            edges[a, b] = edges.get((a, b), 0) + 1
    if any(count != 1 or edges.get((b, a)) != 1 for (a, b), count in edges.items()):
        raise ValueError('boundary is not a closed oriented manifold')
    snapshots, minimum_ratio, minimum_clearance = [], math.inf, math.inf
    for frame in range(101):
        vertices = []
        for node, row in enumerate(rows[frame * node_count:(frame + 1) * node_count]):
            if set(row) != set(FIELDS) or any(value is None for value in row.values()):
                raise ValueError('malformed CSV row')
            if (int(row['frame']), int(row['node'])) != (frame, node):
                raise ValueError('nonconsecutive frame or node')
            values = {key: float(row[key]) for key in FIELDS[1:]}
            if not all(math.isfinite(v) for v in values.values()):
                raise ValueError('nonfinite node state')
            if abs(values['time_s'] - frame * .005) > 1e-12 or abs(values['mass_kg'] - masses[node]) > 5e-8:
                raise ValueError('time or volume-derived nodal mass disagrees')
            vertices.append(tuple(values[k] for k in ('x_m', 'y_m', 'z_m')))
        path = Path(str(prefix) + f'-{frame}.obj')
        payload = path.read_bytes()
        obj_vertices, obj_faces = [], []
        for line in payload.decode().splitlines():
            parts = line.split()
            if parts and parts[0] == 'v':
                obj_vertices.append(tuple(map(float, parts[1:])))
            elif parts and parts[0] == 'f':
                obj_faces.append(tuple(map(int, parts[1:])))
        if vertices != obj_vertices or len(obj_faces) != boundary_count or sorted(map(canonical, obj_faces)) != sorted(map(canonical, faces)):
            raise ValueError('OBJ/CSV node identity or boundary disagreement')
        ratios = [volume([vertices[n] for n in tet]) / rest for tet, rest in zip(corners, volumes)]
        if min(ratios) <= 0 or min(p[2] for p in vertices) < 0:
            raise ValueError('inverted tetrahedron or plane penetration')
        if frame == 0 and max(abs(r - 1) for r in ratios) > 2e-6:
            raise ValueError('reference geometry disagrees with initial vertices')
        minimum_ratio = min(minimum_ratio, *ratios)
        minimum_clearance = min(minimum_clearance, *(p[2] for p in vertices))
        snapshots.append({'frame': frame, 'sha256': hashlib.sha256(payload).hexdigest(),
                          'minimum_volume_ratio': min(ratios)})
    return {'captured_frames': 101, 'rows': len(rows), 'nodes': node_count, 'tetrahedra': element_count,
            'boundary_triangles': boundary_count, 'boundary_oriented_manifold': True,
            'OBJ_CSV_positions_exact': True, 'volume_derived_masses_match': True,
            'minimum_sampled_volume_ratio': minimum_ratio,
            'minimum_sampled_clearance_m': minimum_clearance,
            'trace_sha256': hashlib.sha256(trace).hexdigest(),
            'topology_sha256': hashlib.sha256(topology).hexdigest(),
            'evidence_boundary': 'Exported state geometry and mass only; not independent proof of intermediate device steps, force accuracy, total contact-work closure or fruit calibration.',
            'snapshots': snapshots}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prefix', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    try:
        report = audit(args.prefix)
    except (OSError, ValueError, KeyError, UnicodeError) as error:
        parser.error(str(error))
    output = json.dumps(report, indent=2) + '\n'
    if args.output:
        args.output.write_text(output)
    else:
        print(output, end='')


if __name__ == '__main__':
    main()
