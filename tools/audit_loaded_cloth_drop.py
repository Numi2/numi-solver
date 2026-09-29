#!/usr/bin/env python3
"""Audit the complete authored six-second finite-bench loaded-cloth drop."""

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

from audit_cloth_snapshot import audit, read_snapshot, topology
from audit_replayed_finite_fruit_trace import audit as fruit_audit


def inspect(prefix, release_time, ordinary_mass, hem_mass):
    steps, stride, dt = 720, 10, 1 / 120
    edges, faces, local = topology()
    masses = [hem_mass if 26 * 48 <= node < 28 * 48 else ordinary_mass
              for node in range(1465)]
    rows, contacts = [], []
    for frame in range(0, steps + 1, stride):
        path = Path(f'{prefix}-{frame}.obj')
        payload, vertices, _ = read_snapshot(path, faces)
        grip = [line.split() for line in payload.decode().splitlines()
                if line.startswith('# grip ')]
        if len(grip) != 1 or len(grip[0]) != 15 or grip[0][2] != 'center' or \
                grip[0][6] != 'active' or grip[0][8] != 'orientation' or grip[0][13] != 'patch_center':
            raise ValueError('missing or malformed owning grip state')
        pose = [float(grip[0][field]) for field in (*range(3, 6), *range(9, 13))]
        if not all(math.isfinite(value) for value in pose) or \
                abs(sum(value * value for value in pose[3:]) - 1) > 1e-7:
            raise ValueError('invalid owning grip pose')
        active = int(grip[0][7])
        if active not in (0, 1):
            raise ValueError('invalid grip activation')
        center = [sum(point[axis] * mass for point, mass in zip(vertices, masses)) / sum(masses)
                  for axis in range(3)]
        floor_nodes = [node for node, point in enumerate(vertices)
                       if abs(point[2] + 0.75 - 0.004) <= 2e-6]
        rows.append({'frame': frame, 'time_s': frame * dt, 'snapshot': str(path),
                     'sha256': hashlib.sha256(payload).hexdigest(), 'grip_active': bool(active),
                     'cloth_mass_center_m': center, 'minimum_cloth_node_z_m': min(p[2] for p in vertices),
                     'lower_floor_contact_node_count': len(floor_nodes),
                     'lower_floor_contact_node_mass_kg': sum(masses[node] for node in floor_nodes)})
        contacts.append(audit(path, 0.004, edges, faces, local, include_local=True,
                              require_finite=True, include_yarn_static=True))
    # Snapshots every 83.33 ms do not capture the release at exactly 3.8 s.
    free = [row for row in rows if row['time_s'] >= release_time]
    if len(free) < 2:
        raise ValueError('release does not leave an observable free interval')
    first, final = free[0], free[-1]
    descent = first['cloth_mass_center_m'][2] - final['cloth_mass_center_m'][2]
    first_contact = next((row for row in free if row['lower_floor_contact_node_count']), None)
    inactive = all(not row['grip_active'] for row in free)
    outcome = inactive and descent >= 0.1 and first_contact is not None and final['lower_floor_contact_node_count'] > 0
    trace = fruit_audit(Path(f'{prefix}-fruits.csv'), steps)
    log_path = Path(f'{prefix}.log')
    log_bytes = log_path.read_bytes()
    log = log_bytes.decode()
    terminal = re.findall(r'^result=(PASS|FAIL)$', log, re.MULTILINE)
    model = re.search(r'^model=.*$', log, re.MULTILINE)
    if len(terminal) != 1 or model is None:
        raise ValueError('actual terminal log is required')
    for key, expected in [('scenario', 'recorded'), ('steps', '720'), ('replays', '2'),
                          ('simulated_seconds', '6.000000000')]:
        match = re.search(r'\b' + key + r'=(\S+)', model.group())
        if match is None or match.group(1) != expected:
            raise ValueError(f'terminal log must bind the complete recorded case: {key}')
    exit_path = Path(f'{prefix}.exit')
    actual_exit = int(exit_path.read_text().strip())
    if terminal[0] == 'PASS' and actual_exit != 0 or terminal[0] == 'FAIL' and actual_exit != 1:
        raise ValueError('terminal result and actual exit marker disagree')
    digest = re.search(r'\bframe_state_hash=(0x[0-9a-f]+)\s+replay_frame_states=(\d+)', log)
    replay = 'deterministic=true' in log and digest is not None and int(digest.group(2)) == 721
    contacts_pass = all(row['within_contact_tolerance'] for row in contacts)
    passed = terminal[0] == 'PASS' and actual_exit == 0 and replay and contacts_pass and outcome and \
        trace['serialized_fruit_replay_exact'] and trace['independent_geometry_pass']
    return {'schema': 'numi.finite-bench.loaded-cloth-drop-audit.v1',
            'actual_solver_result': terminal[0], 'actual_exit_code': actual_exit,
            'terminal_log_sha256': hashlib.sha256(log_bytes).hexdigest(),
            'simulated_seconds': 6, 'snapshots': rows, 'snapshot_count': len(rows),
            'mass_model': {'ordinary_node_mass_kg': ordinary_mass, 'hem_node_mass_kg': hem_mass,
                           'total_cloth_mass_kg': sum(masses)},
            'solver_reports_both_ordered_721_frame_replays_match': replay,
            'solver_ordered_frame_hash': digest.group(1) if digest else None,
            'requested_release_time_s': release_time,
            'first_saved_inactive_interval_time_s': first['time_s'],
            'all_post_release_saved_grips_inactive': inactive,
            'cloth_center_descent_after_first_post_release_snapshot_m': descent,
            'first_saved_post_release_lower_floor_contact_time_s': first_contact['time_s'] if first_contact else None,
            'final_lower_floor_contact_node_count': final['lower_floor_contact_node_count'],
            'geometric_loaded_cloth_drop_outcome_pass': outcome,
            'all_saved_contacts_pass': contacts_pass, 'contacts': contacts, 'fruit_trace': trace,
            'qualified_loaded_cloth_drop': passed,
            'boundary': 'Fixed authored 1465-node mass/topology, 4 mm yarn and finite collider. '
              'Caller binds source, binary, trajectory and material. Descent uses mass-weighted saved positions, '
              'and floor contact is geometric, not a support reaction or impact-velocity measurement. '
              'Exact whole-scene replay is reported by the solver; the independent audit compares all fruit CSV states. '
              'Contacts cover saved states, including full yarn interiors, not intervening substeps. '
              'No calibrated material, total contact-work/energy closure or deformable fruit qualification.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prefix', type=Path)
    parser.add_argument('--release-time', type=float, default=3.8)
    parser.add_argument('--ordinary-node-mass', type=float, default=0.00005)
    parser.add_argument('--hem-node-mass', type=float, default=0.0001)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not 0 <= args.release_time < 6 or not all(math.isfinite(x) and x > 0
            for x in (args.ordinary_node_mass, args.hem_node_mass)):
        parser.error('release must precede 6 s and both node masses must be finite and positive')
    try:
        result = inspect(args.prefix, args.release_time, args.ordinary_node_mass, args.hem_node_mass)
    except (OSError, ValueError, IndexError, KeyError) as error:
        parser.error(str(error))
    result['auditor_sha256'] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [Path(__file__), Path(__file__).with_name('audit_cloth_snapshot.py'),
                     Path(__file__).with_name('audit_replayed_finite_fruit_trace.py')]}
    output = json.dumps(result, indent=2) + '\n'
    if args.output:
        args.output.write_text(output)
    else:
        print(output, end='')
    return 0 if result['qualified_loaded_cloth_drop'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
