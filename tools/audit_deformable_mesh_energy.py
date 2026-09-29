#!/usr/bin/env python3
"""Independently reconstruct captured mesh energy and audit accepted work ledgers."""
import argparse
import csv
import hashlib
import json
import math
import struct
from pathlib import Path

from audit_deformable_mesh_trajectory import audit as audit_geometry

FIELDS = ['frame', 'time_s', 'accepted_steps', 'rejected_steps',
          'normal_kinetic_loss_J', 'contact_elastic_change_J',
          'contact_gravity_change_J', 'contact_mechanical_change_J',
          'maximum_positive_contact_step_J', 'free_integration_change_J',
          'absolute_free_integration_change_J', 'maximum_absolute_free_step_J',
          'elastic_force_work_J', 'absolute_position_projection_work_J',
          'maximum_position_projection_step_J', 'maximum_absolute_contact_step_J',
          'positive_contact_work_J', 'independent_mechanical_energy_J']


def sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def fp32(x):
    return struct.unpack('f', struct.pack('f', x))[0]


def audit(prefix, steps=5000):
    geometry = audit_geometry(prefix)
    nodes = geometry['nodes']
    with Path(str(prefix) + '.csv').open() as f:
        trace = list(csv.DictReader(f))
    states = []
    for frame in range(101):
        rows = trace[frame*nodes:(frame+1)*nodes]
        states.append(([tuple(float(r[k]) for k in ('x_m', 'y_m', 'z_m')) for r in rows],
                       [tuple(float(r[k]) for k in ('vx_m_s', 'vy_m_s', 'vz_m_s')) for r in rows],
                       [float(r['mass_kg']) for r in rows]))
    rest = states[0][0]
    elements = []
    with Path(str(prefix) + '-topology.csv').open() as f:
        for row in csv.DictReader(f):
            ids = tuple(int(row[f'node{n}']) for n in range(4))
            a, b, c = (sub(rest[ids[n]], rest[ids[0]]) for n in (1, 2, 3))
            det = dot(a, cross(b, c))
            # Recreate the actual FP32 inverse-rest rows used by the native
            # constitutive law, then evaluate their energy independently in FP64.
            inverse = [tuple(fp32(x / det) for x in r) for r in (cross(b, c), cross(c, a), cross(a, b))]
            elements.append((ids, inverse, float(row['rest_volume_m3']),
                             float(row['mu_Pa']), float(row['lambda_Pa'])))
    energies = []
    gravity = fp32(9.81)
    for positions, velocities, masses in states:
        elastic = 0.
        for ids, inverse, volume, mu, lam in elements:
            edges = [sub(positions[ids[n]], positions[ids[0]]) for n in (1, 2, 3)]
            columns = [tuple(sum(edges[k][r]*inverse[k][col] for k in range(3))
                             for r in range(3)) for col in range(3)]
            J = dot(columns[0], cross(columns[1], columns[2]))
            if not J > 0:
                raise ValueError('inverted captured energy state')
            logJ = math.log(J)
            elastic += volume*(.5*mu*(sum(dot(c, c) for c in columns)-3)-mu*logJ+.5*lam*logJ*logJ)
        kinetic = sum(.5*m*dot(v, v) for m, v in zip(masses, velocities))
        potential = sum(m*gravity*p[2] for m, p in zip(masses, positions))
        energies.append(elastic + kinetic + potential)
    path = Path(str(prefix) + '-energy.csv')
    payload = path.read_bytes()
    with path.open() as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != FIELDS:
            raise ValueError('unexpected energy ledger schema')
        rows = list(reader)
    if len(rows) != 101 or steps not in tuple(5000*2**k for k in range(7)):
        raise ValueError('expected complete 0.5s energy trace and supported timestep')
    maximum_reconstruction = maximum_balance = maximum_components = 0.
    monotonic = ['normal_kinetic_loss_J', 'contact_gravity_change_J',
                 'maximum_positive_contact_step_J', 'absolute_free_integration_change_J',
                 'maximum_absolute_free_step_J', 'absolute_position_projection_work_J',
                 'maximum_position_projection_step_J', 'maximum_absolute_contact_step_J',
                 'positive_contact_work_J']
    previous = {k: 0. for k in monotonic}
    for frame, row in enumerate(rows):
        if set(row) != set(FIELDS) or any(v is None for v in row.values()):
            raise ValueError('malformed energy row')
        values = {k: float(row[k]) for k in FIELDS}
        if not all(math.isfinite(v) for v in values.values()):
            raise ValueError('nonfinite energy ledger')
        if (int(row['frame']), int(row['accepted_steps']), int(row['rejected_steps'])) != (frame, frame*steps//100, 0):
            raise ValueError('nonconsecutive frame or incorrect accepted/rejected step count')
        if abs(values['time_s'] - frame*.005) > 1e-12:
            raise ValueError('energy time does not match nodal trace')
        if frame == 0 and any(values[k] != 0 for k in FIELDS[4:-1]):
            raise ValueError('initial work ledger is not zero')
        for key in monotonic:
            if values[key] < previous[key] or values[key] < 0:
                raise ValueError('negative or nonmonotonic cumulative energy bound')
            previous[key] = values[key]
        if abs(values['free_integration_change_J']) > values['absolute_free_integration_change_J'] + 1e-7:
            raise ValueError('signed integration change exceeds absolute ledger')
        if abs(values['contact_elastic_change_J'])+values['contact_gravity_change_J'] > values['absolute_position_projection_work_J']+1e-6:
            raise ValueError('signed position work exceeds absolute bound')
        maximum_reconstruction = max(maximum_reconstruction,
            abs(energies[frame]-values['independent_mechanical_energy_J']))
        maximum_balance = max(maximum_balance, abs(energies[frame]-energies[0]
            -values['free_integration_change_J']-values['contact_mechanical_change_J']))
        maximum_components = max(maximum_components, abs(values['contact_elastic_change_J']
            +values['contact_gravity_change_J']-values['normal_kinetic_loss_J']
            -values['contact_mechanical_change_J']))
    passed = maximum_reconstruction < 1e-5 and maximum_balance < 1e-4 and maximum_components < 1e-5
    final = {key: float(rows[-1][key]) for key in FIELDS[4:]}
    defect = final['absolute_free_integration_change_J']+final['absolute_position_projection_work_J']
    return {'result': 'PASS' if passed else 'FAIL', 'qualification_kind': 'energy accounting consistency', 'captured_frames': 101,
            'steps': steps, 'nodes': nodes, 'tetrahedra': geometry['tetrahedra'],
            'maximum_FP64_reconstruction_error_J': maximum_reconstruction,
            'maximum_energy_accounting_residual_J': maximum_balance,
            'maximum_contact_component_residual_J': maximum_components,
            'initial_mechanical_energy_J': energies[0], 'final_mechanical_energy_J': energies[-1],
            'energy_error_bound_J': defect, 'energy_error_bound_fraction_of_initial': defect/energies[0],
            'one_percent_numerical_energy_budget_pass': defect <= .01*energies[0],
            'final_work_ledger': final, 'energy_trace_sha256': hashlib.sha256(payload).hexdigest(),
            'nodal_trace_sha256': geometry['trace_sha256'], 'topology_sha256': geometry['topology_sha256'],
            'evidence_boundary': 'Independent captured-state constitutive/kinetic/gravity energy and cumulative ledger consistency. Free integration change is a measured numerical defect, not physical dissipation. Position projection work is reported separately from kinetic loss. Does not certify zero integration error, spatial convergence, calibrated material, reciprocal cloth contact or whole-scene energy closure.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prefix', type=Path)
    parser.add_argument('--steps', type=int, default=5000, choices=tuple(5000*2**k for k in range(7)))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    try:
        result = audit(args.prefix, args.steps)
    except (OSError, ValueError, KeyError, UnicodeError) as error:
        parser.error(str(error))
    text = json.dumps(result, indent=2) + '\n'
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end='')
    return 0 if result['result'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
