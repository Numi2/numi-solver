#!/usr/bin/env python3
"""Audit contacts in exported 48x28/13x13 bag states, without moving any state."""

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

AROUND, LEVELS, GRID = 48, 28, 13


def bottom_index(row, column):
    if row not in (0, GRID - 1) and column not in (0, GRID - 1):
        return AROUND * LEVELS + (row - 1) * 11 + column - 1
    u, v = (column - 6) / 6, (row - 6) / 6
    if abs(u) >= abs(v):
        radius, angle = u, math.pi / 4 * v / u
    else:
        radius, angle = v, math.pi / 2 - math.pi / 4 * u / v
    angle = math.atan2(radius * math.sin(angle), radius * math.cos(angle)) % (2 * math.pi)
    return int(math.floor(angle * AROUND / (2 * math.pi) + 0.5)) % AROUND


def topology():
    boundary = lambda row, column: row in (0, GRID - 1) or column in (0, GRID - 1)
    edges = [(level * AROUND + ring, level * AROUND + (ring + 1) % AROUND)
             for level in range(LEVELS) for ring in range(AROUND)]
    edges += [(level * AROUND + ring, (level + 1) * AROUND + ring)
              for level in range(LEVELS - 1) for ring in range(AROUND)]
    edges += [(bottom_index(row, column), bottom_index(row, column + 1))
              for row in range(GRID) for column in range(GRID - 1)
              if not (boundary(row, column) and boundary(row, column + 1))]
    edges += [(bottom_index(row, column), bottom_index(row + 1, column))
              for column in range(GRID) for row in range(GRID - 1)
              if not (boundary(row, column) and boundary(row + 1, column))]
    faces = []
    for level in range(LEVELS - 1):
        for ring in range(AROUND):
            a, b = level * AROUND + ring, level * AROUND + (ring + 1) % AROUND
            c, d = a + AROUND, b + AROUND
            faces += [(a, b, c), (b, d, c)]
    for row in range(GRID - 1):
        for column in range(GRID - 1):
            a, b = bottom_index(row, column), bottom_index(row, column + 1)
            c, d = bottom_index(row + 1, column), bottom_index(row + 1, column + 1)
            faces += [(a, c, b), (b, c, d)] if (row + column) % 2 == 0 else [(a, d, b), (a, c, d)]
    direct = [set() for _ in range(1465)]
    for first, second in edges:
        direct[first].add(second)
        direct[second].add(first)
    local = [{node} | adjacent | set().union(*(direct[neighbor] for neighbor in adjacent))
             for node, adjacent in enumerate(direct)]
    if len(edges) != 2904 or len(set(edges)) != 2904:
        raise ValueError('unexpected authored topology')
    return edges, faces, local


def subtract(first, second):
    return tuple(a - b for a, b in zip(first, second))


def dot(first, second):
    return sum(a * b for a, b in zip(first, second))


def clamp(value):
    return max(0.0, min(1.0, value))


def point_segment_distance(point, first, second):
    direction, offset = subtract(second, first), subtract(point, first)
    length_squared = dot(direction, direction)
    weight = clamp(dot(offset, direction) / length_squared) if length_squared else 0.0
    return math.sqrt(sum((p - a - weight * d) ** 2 for p, a, d in zip(point, first, direction)))


def segment_distance(first, second, third, fourth):
    u, v, w = subtract(second, first), subtract(fourth, third), subtract(first, third)
    a, b, c, d, e = dot(u, u), dot(u, v), dot(v, v), dot(u, w), dot(v, w)
    if a == 0:
        return point_segment_distance(first, third, fourth)
    if c == 0:
        return point_segment_distance(third, first, second)
    denominator = a * c - b * b
    s = clamp((b * e - c * d) / denominator) if denominator > 1e-14 * a * c else 0.0
    t = (b * s + e) / c
    if t < 0:
        t, s = 0.0, clamp(-d / a)
    elif t > 1:
        t, s = 1.0, clamp((b - d) / a)
    return math.sqrt(sum((x + s * y - t * z) ** 2 for x, y, z in zip(w, u, v)))


def self_overlap(vertices, edges, local, radius):
    # Inflated segment AABBs cover every possible capsule overlap. Sharing a
    # grid cell supplies candidates only; exact segment distance decides contact.
    cell_size = max(0.032, 4 * radius)
    cells, pairs = {}, set()
    for index, (first, second) in enumerate(edges):
        bounds = [(math.floor((min(a, b) - radius) / cell_size),
                   math.floor((max(a, b) + radius) / cell_size))
                  for a, b in zip(vertices[first], vertices[second])]
        count = math.prod(high - low + 1 for low, high in bounds)
        if count > 100000:
            raise ValueError('unbounded yarn AABB; snapshot is outside the auditor contract')
        for cell in itertools.product(*(range(low, high + 1) for low, high in bounds)):
            previous = cells.setdefault(cell, [])
            pairs.update((other, index) for other in previous)
            previous.append(index)
    maximum, witness, eligible = 0.0, None, 0
    for first, second in sorted(pairs):
        a, b = edges[first]
        c, d = edges[second]
        if c in local[a] or d in local[a] or c in local[b] or d in local[b]:
            continue
        eligible += 1
        overlap = 2 * radius - segment_distance(vertices[a], vertices[b], vertices[c], vertices[d])
        if overlap > maximum:
            maximum, witness = overlap, [list(edges[first]), list(edges[second])]
    return maximum, witness, eligible


def read_snapshot(path, expected_faces):
    """Read an exact authored render topology without interpreting its physics."""
    payload = path.read_bytes()
    vertices, faces, fruits = [], [], {}
    for line in payload.decode('utf-8').splitlines():
        fields = line.split()
        if fields and fields[0] == 'v':
            if len(fields) != 4:
                raise ValueError('unsupported vertex encoding')
            vertices.append(tuple(map(float, fields[1:4])))
        elif fields and fields[0] == 'f':
            faces.append(tuple(int(value) - 1 for value in fields[1:]))
        elif fields[:2] == ['#', 'ball']:
            index = int(fields[2])
            if index in fruits or fields[3] != 'center' or fields[7] != 'radius':
                raise ValueError('invalid or duplicate fruit')
            fruits[index] = (tuple(map(float, fields[4:7])), float(fields[8]))
    if len(vertices) != 1465 or faces != expected_faces or set(fruits) != set(range(12)):
        raise ValueError('snapshot does not have the supported authored bag topology')
    if not all(math.isfinite(value) for point in vertices for value in point):
        raise ValueError('nonfinite cloth position')
    if not all(math.isfinite(value) for center, r in fruits.values() for value in (*center, r)):
        raise ValueError('nonfinite fruit geometry')
    if any(r <= 0 for _, r in fruits.values()):
        raise ValueError('nonpositive fruit radius')
    return payload, vertices, fruits


def local_node_pairs(edges, local):
    direct = [set() for _ in local]
    for first, second in edges:
        direct[first].add(second)
        direct[second].add(first)
    pairs = [(first, second) for first, neighbors in enumerate(local)
             for second in sorted(neighbors) if second > first and second not in direct[first]]
    if len(pairs) != 5754 or (861, 957) not in pairs:
        raise ValueError('unexpected two-hop node contact topology')
    return pairs


def static_geometry(payload, require_finite=False):
    declarations = [line.split() for line in payload.decode('utf-8').splitlines()
                    if line.split()[:2] == ['#', 'static_bench']]
    if not declarations:
        if require_finite:
            raise ValueError('required finite bench declaration is missing')
        return 'unbounded_plane'
    expected = ['#', 'static_bench', 'min', '-0.75', '-0.5', '-0.08',
                'max', '0.75', '0.5', '0', 'floor', '-0.75']
    if len(declarations) != 1 or declarations[0] != expected:
        raise ValueError('static bench does not match the authored finite collider')
    return 'finite_bench_and_room_floor'


def surface_gap(point, radius, geometry):
    if geometry == 'unbounded_plane':
        return point[2] - radius
    # Independent centered-box SDF; no solver helper or sweep is imported.
    q = (abs(point[0]) - 0.75, abs(point[1]) - 0.5, abs(point[2] + 0.04) - 0.04)
    box_gap = math.hypot(*(max(0.0, value) for value in q)) + min(max(q), 0.0) - radius
    return min(box_gap, point[2] + 0.75 - radius)


def audit(path, radius, edges, expected_faces, local, include_local=False, require_finite=False):
    payload, vertices, fruits = read_snapshot(path, expected_faces)
    geometry = static_geometry(payload, require_finite)
    fruit_overlap, fruit_witness = 0.0, None
    for fruit, (center, r) in fruits.items():
        for edge in edges:
            overlap = r + radius - point_segment_distance(center, vertices[edge[0]], vertices[edge[1]])
            if overlap > fruit_overlap:
                fruit_overlap, fruit_witness = overlap, {'fruit': fruit, 'yarn_endpoints': list(edge)}
    pair_overlap, pair_witness = 0.0, None
    for first, second in itertools.combinations(sorted(fruits), 2):
        a, ra = fruits[first]
        b, rb = fruits[second]
        overlap = ra + rb - math.dist(a, b)
        if overlap > pair_overlap:
            pair_overlap, pair_witness = overlap, [first, second]
    self_penetration, self_witness, candidates = self_overlap(vertices, edges, local, radius)
    ground = max(0.0, max(-surface_gap(vertex, radius, geometry) for vertex in vertices),
                 max(-surface_gap(center, r, geometry) for center, r in fruits.values()))
    row = {'snapshot': str(path), 'snapshot_sha256': hashlib.sha256(payload).hexdigest(),
            'static_surface_model': geometry,
            'fruit_yarn_overlap_m': fruit_overlap, 'fruit_yarn_witness': fruit_witness,
            'fruit_pair_overlap_m': pair_overlap, 'fruit_pair_witness': pair_witness,
            'nonlocal_yarn_overlap_m': self_penetration, 'nonlocal_yarn_witness': self_witness,
            'self_broadphase_candidates': candidates, 'ground_overlap_m': ground,
            'within_contact_tolerance': max(fruit_overlap, pair_overlap, self_penetration, ground) <= 2e-6}
    if include_local:
        pairs = local_node_pairs(edges, local)
        maximum, witness = 0.0, None
        for first, second in pairs:
            overlap = 2 * radius - math.dist(vertices[first], vertices[second])
            if overlap > maximum:
                maximum, witness = overlap, [first, second]
        row.update(local_node_pair_count=len(pairs), local_node_overlap_m=maximum,
                   local_node_witness=witness)
        row['within_contact_tolerance'] &= maximum <= 2e-6
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshots', nargs='+', type=Path)
    parser.add_argument('--yarn-radius', type=float, default=0.004,
                        help='Radius from the source/material manifest; default authored 0.004 m.')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--require-finite-bench', action='store_true',
                        help='Reject a missing or mismatched authored finite bench declaration.')
    parser.add_argument('--include-local-node-contacts', action='store_true',
                        help='Enforce the ABI 14 two-hop non-direct node diameter contacts. '
                             'Leave unset only when auditing retained pre-repair artifacts.')
    args = parser.parse_args()
    if not math.isfinite(args.yarn_radius) or args.yarn_radius <= 0:
        parser.error('yarn radius must be finite and positive')
    try:
        edges, faces, local = topology()
        rows = [audit(path, args.yarn_radius, edges, faces, local, args.include_local_node_contacts,
                      args.require_finite_bench)
                for path in args.snapshots]
    except (OSError, ValueError, IndexError, UnicodeError) as error:
        parser.error(str(error))
    report = {'yarn_radius_m': args.yarn_radius, 'yarn_count': len(edges),
              'local_node_contacts_enforced': args.include_local_node_contacts,
              'all_snapshots_within_contact_tolerance': all(row['within_contact_tolerance'] for row in rows),
              'evidence_boundary': 'Contacts of supplied exported states only. Exact authored axial graph and two-hop exclusions are reconstructed after validating every render triangle. Does not certify intervening substeps, dynamics, material calibration, strain or replay.',
              'snapshots': rows}
    output = json.dumps(report, indent=2) + '\n'
    if args.output:
        args.output.write_text(output)
    else:
        print(output, end='')
    return 0 if report['all_snapshots_within_contact_tolerance'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
