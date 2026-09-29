#!/usr/bin/env python3
"""Report fruit location relative to the exported bag mesh and a virtual mouth cap."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from audit_cloth_snapshot import AROUND, LEVELS, dot, read_snapshot, subtract, topology


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def closed_faces(faces):
    # The authored wall/bottom faces are outward-oriented. Close only the
    # open top boundary with an oriented fan; this cap is diagnostic geometry.
    first = AROUND * (LEVELS - 1)
    center = 1465
    closed = faces + [(first + ring, first + (ring + 1) % AROUND, center)
                      for ring in range(AROUND)]
    directed = {}
    for triangle in closed:
        for a, b in zip(triangle, triangle[1:] + triangle[:1]):
            directed[a, b] = directed.get((a, b), 0) + 1
    if any(count != 1 or directed.get((b, a)) != 1
           for (a, b), count in directed.items()):
        raise ValueError('virtual cap did not close an oriented manifold')
    return closed


def winding_number(point, vertices, faces):
    angles = []
    for indices in faces:
        a, b, c = (subtract(vertices[index], point) for index in indices)
        la, lb, lc = (math.sqrt(dot(v, v)) for v in (a, b, c))
        numerator = dot(a, cross(b, c))
        denominator = la * lb * lc + dot(a, b) * lc + dot(b, c) * la + dot(c, a) * lb
        if min(la, lb, lc) == 0 or (numerator == 0 and denominator <= 0):
            return None  # Point on a diagnostic triangle: no stable inside/outside claim.
        angles.append(2 * math.atan2(numerator, denominator))
    return math.fsum(angles) / (4 * math.pi)


def audit(path, faces, closed):
    payload, vertices, fruits = read_snapshot(path, faces)
    ring = vertices[AROUND * (LEVELS - 1):AROUND * LEVELS]
    vertices.append(tuple(math.fsum(p[k] for p in ring) / AROUND for k in range(3)))
    rows = []
    for index, (center, radius) in sorted(fruits.items()):
        winding = winding_number(center, vertices, closed)
        magnitude = abs(winding) if winding is not None else None
        location = ('ambiguous' if magnitude is None else
                    'outside' if magnitude < 1e-6 else
                    'inside' if abs(magnitude - 1) < 1e-6 else 'ambiguous')
        rows.append({'fruit': index, 'render_mesh_winding_number': winding,
                     'center_location': location,
                     'ground_clearance_m': center[2] - radius})
    return {'snapshot': str(path), 'snapshot_sha256': hashlib.sha256(payload).hexdigest(),
            'fruits': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshots', nargs='+', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    try:
        _, faces, _ = topology()
        closed = closed_faces(faces)
        rows = [audit(path, faces, closed) for path in args.snapshots]
    except (OSError, ValueError, IndexError, UnicodeError) as error:
        parser.error(str(error))
    result = {'snapshots': rows, 'evidence_boundary':
              'Fruit centers relative to the exact render mesh closed by a virtual top-mouth fan. '
              'The cap is not a collision surface. Winding does not prove full-sphere clearance, '
              'an intervening release or re-entry, calibrated material, dynamics or solver replay. '
              'Self-intersecting geometry or a point on the cap can be ambiguous. '
              'This descriptive report has no physical qualification gate.'}
    output = json.dumps(result, indent=2) + '\n'
    if args.output:
        args.output.write_text(output)
    else:
        print(output, end='')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
