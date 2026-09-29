#!/usr/bin/env python3
"""Compare matched native meshes without changing the authored body or motion."""
import argparse
import csv
import io
import json
import math
from pathlib import Path
from audit_deformable_mesh_trajectory import audit


def load(prefix):
    certificate = audit(prefix)
    rows = list(csv.DictReader(io.StringIO(Path(str(prefix) + '.csv').read_text())))
    count = certificate['nodes']
    frames = [rows[f * count:(f + 1) * count] for f in range(101)]
    positions = [[tuple(float(row[k]) for k in ('x_m', 'y_m', 'z_m'))
                  for row in frame] for frame in frames]
    masses = [float(row['mass_kg']) for row in frames[0]]
    total_mass = sum(masses)
    centers = [tuple(sum(p[k] * m for p, m in zip(frame, masses)) / total_mass
                     for k in range(3)) for frame in positions]
    return certificate, positions, centers, total_mass


def compare(prefixes, tolerance):
    levels = [load(prefix) for prefix in prefixes]
    supported = [13, 55, 309, 2057, 14993]
    counts = [level[0]['nodes'] for level in levels]
    if len(counts) < 2 or counts[0] not in supported or \
            counts != supported[supported.index(counts[0]):supported.index(counts[0]) + len(counts)]:
        raise ValueError('provide at least two consecutive supported levels in increasing order')
    comparison = []
    height = lambda frame: max(p[2] for p in frame) - min(p[2] for p in frame)
    for index in range(len(levels) - 1):
        a, coarse, ca, ma = levels[index]
        b, fine, cb, mb = levels[index + 1]
        if coarse[0] != fine[0][:len(coarse[0])] or abs(ma - mb) > 2e-7:
            raise ValueError('refinement changed pre-existing rest nodes or total mass')
        node_error = max(math.dist(p, q) for fa, fb in zip(coarse, fine)
                         for p, q in zip(fa, fb[:len(fa)]))
        comparison.append({
            'coarse_nodes': a['nodes'], 'fine_nodes': b['nodes'],
            'coarse_tetrahedra': a['tetrahedra'], 'fine_tetrahedra': b['tetrahedra'],
            'total_mass_difference_kg': abs(ma - mb),
            'maximum_matched_node_difference_m': node_error,
            'maximum_COM_difference_m': max(math.dist(p, q) for p, q in zip(ca, cb)),
            'maximum_height_difference_m': max(abs(height(fa) - height(fb))
                                               for fa, fb in zip(coarse, fine)),
            'within_spatial_target': node_error <= tolerance,
            'coarse_trace_sha256': a['trace_sha256'], 'fine_trace_sha256': b['trace_sha256'],
        })
    return {'spatial_target_m': tolerance, 'comparison': comparison,
            'finest_pair_within_spatial_target': comparison[-1]['within_spatial_target'],
            'evidence_boundary': 'Matched owning-node positions at equal exported times for the same '
            'piecewise-flat rest body, authored material and volume-derived masses. Each complete '
            'trajectory is independently checked for topology, geometry and mass. Timestep convergence '
            'does not imply spatial convergence. This comparison does not calibrate fruit properties '
            'or close intervening contact work.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prefixes', nargs='+', type=Path)
    parser.add_argument('--spatial-target-m', type=float, default=.001)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not math.isfinite(args.spatial_target_m) or args.spatial_target_m <= 0:
        parser.error('spatial target must be finite and positive')
    try:
        report = compare(args.prefixes, args.spatial_target_m)
    except (OSError, ValueError, KeyError, UnicodeError) as error:
        parser.error(str(error))
    payload = json.dumps(report, indent=2) + '\n'
    if args.output:
        args.output.write_text(payload)
    else:
        print(payload, end='')
    return 0 if report['finest_pair_within_spatial_target'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
