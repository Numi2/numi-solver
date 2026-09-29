#!/usr/bin/env python3
"""Check both CPU/native fruit replays and independently reconstruct finite-bench gaps."""

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path

from audit_cloth_snapshot import surface_gap

COLUMNS = ('replay,frame,time_s,fruit,x_m,y_m,z_m,radius_m,vx_m_s,vy_m_s,vz_m_s,'
           'wx_rad_s,wy_rad_s,wz_rad_s,static_clearance_m,released').split(',')


def audit(path, expected_frames):
    payload = path.read_bytes()
    reader = csv.DictReader(io.StringIO(payload.decode('utf-8')))
    columns = reader.fieldnames
    native = columns == COLUMNS + ['last_substep_ground_impulse_Ns']
    if columns != COLUMNS and not native:
        raise ValueError('expected exact replayed CPU or native finite-bench fruit columns')
    replays = {1: {}, 2: {}}
    discrepancy = 0.0
    minimum_gap = math.inf
    for row in reader:
        if set(row) != set(columns) or any(value is None for value in row.values()):
            raise ValueError('incomplete or malformed trace row; this is not terminal evidence')
        replay, frame, fruit = (int(row[key]) for key in ('replay', 'frame', 'fruit'))
        if replay not in replays or not 0 <= frame <= expected_frames or not 0 <= fruit < 12:
            raise ValueError('invalid replay, frame or fruit identity')
        state = {key: float(row[key]) for key in columns[2:] if key != 'fruit'}
        if not all(math.isfinite(value) for value in state.values()):
            raise ValueError('nonfinite fruit state')
        if state['radius_m'] <= 0 or state['released'] not in (0, 1):
            raise ValueError('invalid radius or release bit')
        if native and state['last_substep_ground_impulse_Ns'] < 0:
            raise ValueError('negative native static normal impulse')
        key = frame, fruit
        if key in replays[replay]:
            raise ValueError('duplicate fruit state')
        replays[replay][key] = state
        point = tuple(state[f'{axis}_m'] for axis in 'xyz')
        gap = surface_gap(point, state['radius_m'], 'finite_bench_and_room_floor')
        discrepancy = max(discrepancy, abs(gap - state['static_clearance_m']))
        minimum_gap = min(minimum_gap, gap)
    complete_keys = {(frame, fruit) for frame in range(expected_frames + 1) for fruit in range(12)}
    complete = all(set(replay) == complete_keys for replay in replays.values())
    if not complete:
        raise ValueError('both complete replays are required; live prefixes are not qualification')
    dt = replays[1][1, 0]['time_s']
    if dt <= 0:
        raise ValueError('nonpositive frame timestep')
    summaries = []
    for replay, states in replays.items():
        fruits = []
        for fruit in range(12):
            history = [states[frame, fruit] for frame in range(expected_frames + 1)]
            radius = history[0]['radius_m']
            release_frame = floor_frame = None
            released = False
            downward = 0.0
            for frame, state in enumerate(history):
                if abs(state['time_s'] - frame * dt) > 2e-12 or state['radius_m'] != radius:
                    raise ValueError('frame time or fruit radius changed between states')
                if released and not state['released']:
                    raise ValueError('latched release was cleared')
                if state['released']:
                    if not released:
                        release_frame = frame
                    released = True
                    downward = max(downward, -state['vz_m_s'])
                    floor_gap = state['z_m'] + 0.75 - radius
                    if floor_frame is None and abs(floor_gap) <= 2e-6 and abs(state['vz_m_s']) <= 1e-3:
                        floor_frame = frame
            final = history[-1]
            floor_gap = final['z_m'] + 0.75 - radius
            fruits.append({'fruit': fruit, 'first_release_frame': release_frame,
                'first_release_time_s': None if release_frame is None else release_frame * dt,
                'first_lower_floor_contact_frame_after_release': floor_frame,
                'first_lower_floor_contact_time_s_after_release': None if floor_frame is None else floor_frame * dt,
                'maximum_downward_speed_after_release_m_s': downward,
                'final_center_m': [final[f'{axis}_m'] for axis in 'xyz'],
                'final_static_gap_m': final['static_clearance_m'],
                'final_floor_gap_m': floor_gap, 'final_vertical_velocity_m_s': final['vz_m_s'],
                'final_lower_floor_contact': abs(floor_gap) <= 2e-6 and abs(final['vz_m_s']) <= 1e-3})
        released_fruits = [row['fruit'] for row in fruits if row['first_release_frame'] is not None]
        landed = [row['fruit'] for row in fruits if row['first_release_frame'] is not None and
                  row['first_lower_floor_contact_frame_after_release'] is not None and row['final_lower_floor_contact']]
        summaries.append({'replay': replay, 'frames': expected_frames + 1,
                          'released_fruits': released_fruits, 'released_fruits_landed_on_floor': landed,
                          'fruits': fruits})
    exact = replays[1] == replays[2]
    geometry = minimum_gap >= -2e-6 and discrepancy <= 1e-9
    return {'schema': 'numi.finite-bench.fruit-trace-audit.v1', 'trace': str(path),
        'trace_sha256': hashlib.sha256(payload).hexdigest(), 'simulated_seconds': expected_frames * dt,
        'backend': 'native_fp32' if native else 'cpu_fp64',
        'both_replays_complete': complete, 'serialized_fruit_replay_exact': exact,
        'minimum_independent_static_gap_m': minimum_gap,
        'maximum_exported_gap_discrepancy_m': discrepancy, 'independent_geometry_pass': geometry,
        'replays': summaries,
        'evidence_boundary': 'The caller asserts the authored finite-bench geometry and binds source/config separately. '
          'Both complete CSV fruit sequences are compared and gaps independently reconstructed. Floor support is '
          'an observed radius/vertical-velocity match, not a reaction-force audit. Release is latched and may include '
          're-entry. This does not establish whole-cloth replay, cloth contacts, intervening substeps, calibration or energy closure.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace', type=Path)
    parser.add_argument('--expected-frames', type=int, default=480)
    parser.add_argument('--require-released-landings', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.expected_frames < 1:
        parser.error('expected frame count must be positive')
    try:
        result = audit(args.trace, args.expected_frames)
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.error(str(error))
    passed = result['serialized_fruit_replay_exact'] and result['independent_geometry_pass']
    if args.require_released_landings:
        passed &= all(len(row['released_fruits']) >= 2 and
                      row['released_fruits'] == row['released_fruits_landed_on_floor']
                      for row in result['replays'])
    result['requested_checks_pass'] = passed
    output = json.dumps(result, indent=2) + '\n'
    if args.output:
        args.output.write_text(output)
    else:
        print(output, end='')
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
