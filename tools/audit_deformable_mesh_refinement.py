#!/usr/bin/env python3
"""Compare matched native meshes without changing the authored body or motion."""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
from audit_deformable_mesh_trajectory import audit
from audit_deformable_mesh_energy import FIELDS as ENERGY_FIELDS, VERLET_FIELDS


def load_integration(prefix):
    """Bind a spatial comparison to its actual accepted temporal discretization."""
    path = Path(str(prefix) + '-energy.csv')
    if not path.exists():
        raise ValueError('spatial comparison requires the owning energy ledger to identify '
                         'the integrator and accepted step grid')
    payload = path.read_bytes()
    reader = csv.DictReader(io.StringIO(payload.decode()))
    fields = reader.fieldnames
    if fields not in (ENERGY_FIELDS, VERLET_FIELDS):
        raise ValueError('unexpected energy ledger schema in spatial comparison')
    rows = list(reader)
    if len(rows) != 101:
        raise ValueError('spatial comparison requires 101 complete energy states')
    if any(set(row) != set(fields) or any(v is None for v in row.values()) for row in rows):
        raise ValueError('malformed energy row in spatial comparison')
    extended = fields == VERLET_FIELDS
    integrator = int(rows[0]['integrator']) if extended else 0
    if integrator not in (0, 1):
        raise ValueError('unsupported spatial comparison integrator')
    steps = int(rows[-1]['accepted_steps'])
    if steps not in tuple(5000 * 2**k for k in range(7)):
        raise ValueError('unsupported accepted step grid in spatial comparison')
    for frame, row in enumerate(rows):
        values = {key: float(row[key]) for key in fields}
        if not all(math.isfinite(v) for v in values.values()):
            raise ValueError('nonfinite energy ledger in spatial comparison')
        if (values['frame'], values['accepted_steps'], values['rejected_steps']) != \
                (frame, frame * steps // 100, 0) or \
                abs(values['time_s'] - frame * .005) > 1e-12:
            raise ValueError('energy frames or accepted steps disagree with the complete nodal trace')
        if extended and values['integrator'] != integrator:
            raise ValueError('integrator changes within spatial comparison trace')
        if values['normal_kinetic_loss_J'] < 0:
            raise ValueError('negative normal kinetic loss in spatial comparison')
    return {'integrator': ('euler', 'support-verlet')[integrator],
            'accepted_steps': steps, 'ledger_sha256': hashlib.sha256(payload).hexdigest(),
            'first_captured_impact_frame': next((frame for frame, row in enumerate(rows)
                if float(row['normal_kinetic_loss_J']) > 0), None),
            'final_normal_kinetic_loss_J': float(rows[-1]['normal_kinetic_loss_J'])}


def load(prefix):
    certificate = audit(prefix)
    integration = load_integration(prefix)
    rows = list(csv.DictReader(io.StringIO(Path(str(prefix) + '.csv').read_text())))
    count = certificate['nodes']
    frames = [rows[f * count:(f + 1) * count] for f in range(101)]
    positions = [[tuple(float(row[k]) for k in ('x_m', 'y_m', 'z_m'))
                  for row in frame] for frame in frames]
    masses = [float(row['mass_kg']) for row in frames[0]]
    initial_velocities = [tuple(float(row[k]) for k in ('vx_m_s', 'vy_m_s', 'vz_m_s'))
                          for row in frames[0]]
    if any(v != (0., 0., 0.) for v in initial_velocities):
        raise ValueError('refinement changed the authored initially stationary drop')
    total_mass = sum(masses)
    centers = [tuple(sum(p[k] * m for p, m in zip(frame, masses)) / total_mass
                     for k in range(3)) for frame in positions]
    return certificate, positions, centers, total_mass, integration


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
        a, coarse, ca, ma, ia = levels[index]
        b, fine, cb, mb, ib = levels[index + 1]
        if coarse[0] != fine[0][:len(coarse[0])] or abs(ma - mb) > 2e-7:
            raise ValueError('refinement changed pre-existing rest nodes or total mass')
        if ia['integrator'] != ib['integrator']:
            raise ValueError('spatial comparison requires the same integration method')
        if ia['accepted_steps'] != ib['accepted_steps']:
            raise ValueError('spatial comparison requires the same accepted step grid; '
                             'mixed timestep and mesh changes are not spatial convergence')
        frame_errors = [max((math.dist(p, q), node) for node, (p, q) in
                           enumerate(zip(fa, fb[:len(fa)]))) for fa, fb in zip(coarse, fine)]
        node_error, maximum_frame, maximum_node = max((error, frame, node)
            for frame, (error, node) in enumerate(frame_errors))
        impact_frames = [i['first_captured_impact_frame'] for i in (ia, ib)
                         if i['first_captured_impact_frame'] is not None]
        first_impact = min(impact_frames) if impact_frames else len(coarse)
        comparison.append({
            'coarse_nodes': a['nodes'], 'fine_nodes': b['nodes'],
            'coarse_tetrahedra': a['tetrahedra'], 'fine_tetrahedra': b['tetrahedra'],
            'total_mass_difference_kg': abs(ma - mb),
            'maximum_matched_node_difference_m': node_error,
            'maximum_difference_frame': maximum_frame,
            'maximum_difference_time_s': maximum_frame * .005,
            'maximum_difference_node': maximum_node,
            'maximum_pre_captured_impact_node_difference_m':
                max((error for error, _ in frame_errors[:first_impact]), default=0.),
            'coarse_first_captured_impact_frame': ia['first_captured_impact_frame'],
            'fine_first_captured_impact_frame': ib['first_captured_impact_frame'],
            'coarse_normal_kinetic_loss_J': ia['final_normal_kinetic_loss_J'],
            'fine_normal_kinetic_loss_J': ib['final_normal_kinetic_loss_J'],
            'integrator': ia['integrator'], 'accepted_steps': ia['accepted_steps'],
            'maximum_COM_difference_m': max(math.dist(p, q) for p, q in zip(ca, cb)),
            'maximum_height_difference_m': max(abs(height(fa) - height(fb))
                                               for fa, fb in zip(coarse, fine)),
            'within_spatial_target': node_error <= tolerance,
            'coarse_trace_sha256': a['trace_sha256'], 'fine_trace_sha256': b['trace_sha256'],
            'coarse_energy_ledger_sha256': ia['ledger_sha256'],
            'fine_energy_ledger_sha256': ib['ledger_sha256'],
        })
    return {'spatial_target_m': tolerance, 'comparison': comparison,
            'finest_pair_within_spatial_target': comparison[-1]['within_spatial_target'],
            'evidence_boundary': 'Matched owning-node positions at equal exported times for the same '
            'piecewise-flat rest body, authored material and volume-derived masses. Each complete '
            'trajectory is independently checked for topology, geometry and mass. Owning energy '
            'ledgers bind equal integrators and accepted step grids, and both meshes start stationary. '
            'Captured impact times delimit the observed pre-impact region; they are not exact '
            'continuous impact times. Timestep convergence '
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
