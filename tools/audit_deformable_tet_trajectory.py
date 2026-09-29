#!/usr/bin/env python3
"""Independently check retained native tetrahedron CSV and OBJ states."""

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path


FIELDS = ['frame', 'time_s', 'element', 'node', 'x_m', 'y_m', 'z_m',
          'vx_m_s', 'vy_m_s', 'vz_m_s', 'mass_kg']


def volume(points):
    a, b, c = [tuple(points[n][k] - points[0][k] for k in range(3))
               for n in (1, 2, 3)]
    return (a[0] * (b[1] * c[2] - b[2] * c[1]) +
            a[1] * (b[2] * c[0] - b[0] * c[2]) +
            a[2] * (b[0] * c[1] - b[1] * c[0])) / 6


def audit(prefix):
    trace = Path(str(prefix) + '.csv')
    payload = trace.read_bytes()
    reader = csv.DictReader(io.StringIO(payload.decode('utf-8')))
    if reader.fieldnames != FIELDS:
        raise ValueError('unexpected trace schema')
    rows = list(reader)
    if len(rows) != 51 * 12:
        raise ValueError('expected 51 complete states of three four-node elements')
    faces = [tuple(4 * element + n for n in face)
             for element in range(3)
             for face in ((1, 3, 2), (1, 2, 4), (1, 4, 3), (2, 3, 4))]
    masses = {}
    minimum_volume = math.inf
    snapshots = []
    for frame in range(51):
        state = rows[frame * 12:(frame + 1) * 12]
        vertices = []
        for index, row in enumerate(state):
            if (int(row['frame']), int(row['element']), int(row['node'])) != (
                    frame, index // 4, index % 4):
                raise ValueError('nonconsecutive frame, element or node')
            values = {key: float(row[key]) for key in FIELDS[1:]}
            if not all(math.isfinite(value) for value in values.values()):
                raise ValueError('nonfinite trajectory state')
            if abs(values['time_s'] - frame * .01) > 1e-12:
                raise ValueError('incorrect capture time')
            mass = values['mass_kg']
            if mass <= 0 or (index in masses and masses[index] != mass):
                raise ValueError('nonpositive or changed nodal mass')
            masses[index] = mass
            vertices.append(tuple(values[key] for key in ('x_m', 'y_m', 'z_m')))
        path = Path(str(prefix) + f'-{frame}.obj')
        obj = path.read_bytes()
        obj_vertices, obj_faces = [], []
        for line in obj.decode('utf-8').splitlines():
            fields = line.split()
            if fields and fields[0] == 'v':
                if len(fields) != 4:
                    raise ValueError('unexpected OBJ vertex encoding')
                obj_vertices.append(tuple(map(float, fields[1:])))
            elif fields and fields[0] == 'f':
                obj_faces.append(tuple(map(int, fields[1:])))
        if obj_vertices != vertices or obj_faces != faces:
            raise ValueError('OBJ geometry disagrees with CSV or tetrahedron boundary')
        volumes = [volume(vertices[n:n + 4]) for n in range(0, 12, 4)]
        if min(volumes) <= 0:
            raise ValueError('nonpositive exported element volume')
        minimum_volume = min(minimum_volume, *volumes)
        snapshots.append({'frame': frame, 'sha256': hashlib.sha256(obj).hexdigest(),
                          'signed_volumes_m3': volumes})
    return {'trace_sha256': hashlib.sha256(payload).hexdigest(),
            'captured_frames': 51, 'elements': 3, 'rows': len(rows),
            'simulated_seconds': .5, 'OBJ_CSV_positions_exact': True,
            'all_sampled_volumes_positive': True,
            'minimum_sampled_volume_m3': minimum_volume,
            'evidence_boundary': 'Exported states only; does not independently verify intervening GPU steps, contact, material calibration or replay.',
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
    result = json.dumps(report, indent=2) + '\n'
    if args.output:
        args.output.write_text(result)
    else:
        print(result, end='')


if __name__ == '__main__':
    main()
