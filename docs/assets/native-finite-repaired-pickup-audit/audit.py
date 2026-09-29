#!/usr/bin/env python3
"""Read-only, terminal-only audit of the frozen 2x480 Metal finite-bench pickup.

This does not launch Metal or treat sampled exports as internal frame-hash proof.
"""
import csv
import hashlib
import io
import json
import math
import os
import pathlib
import re
import struct
import sys

ROOT = pathlib.Path('/Users/home/numi-solver')
PREFIX = ROOT / 'build/native-finite-repaired-pickup-20260929'
OUTPUT = pathlib.Path(__file__).with_name('result.json')
HEADER = ('replay,frame,time_s,fruit,x_m,y_m,z_m,radius_m,vx_m_s,vy_m_s,vz_m_s,'
          'wx_rad_s,wy_rad_s,wz_rad_s,static_clearance_m,released,last_substep_ground_impulse_Ns')
FRAMES = 480
FRAME_DT = 48 * struct.unpack('<f', struct.pack('<f', 1 / 5760))[0]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(condition, message):
    if not condition:
        raise ValueError(message)


def finite_bench_gap(x, y, z, radius):
    q = (abs(x) - .75, abs(y) - .5, abs(z + .04) - .04)
    box = math.hypot(*(max(0., value) for value in q)) + min(max(q), 0.) - radius
    return min(box, z + .75 - radius)


def trace(path, replay):
    payload = path.read_bytes()
    reader = csv.DictReader(io.StringIO(payload.decode()))
    check(reader.fieldnames == HEADER.split(','), f'{path.name}: columns')
    rows = list(reader)
    check(len(rows) == (FRAMES + 1) * 12, f'{path.name}: incomplete rows')
    parsed = {}
    minimum_gap = math.inf
    maximum_gap_error = 0.
    radii = {}
    release_first = {}
    release_last = set()
    for offset, row in enumerate(rows):
        frame, fruit = divmod(offset, 12)
        check(int(row['replay']) == replay and int(row['frame']) == frame and
              int(row['fruit']) == fruit, f'{path.name}: row order/identity {offset}')
        values = {key: float(value) for key, value in row.items() if key not in ('replay', 'frame', 'fruit')}
        check(all(math.isfinite(value) for value in values.values()), f'{path.name}: nonfinite {offset}')
        check(abs(values['time_s'] - frame * FRAME_DT) <= 2e-12, f'{path.name}: time {offset}')
        radius = values['radius_m']
        check(radius > 0 and (fruit not in radii or radii[fruit] == radius), f'{path.name}: radius {offset}')
        radii[fruit] = radius
        check(values['last_substep_ground_impulse_Ns'] >= 0, f'{path.name}: negative ground impulse {offset}')
        released = values['released']
        check(released in (0, 1), f'{path.name}: release bit {offset}')
        if released:
            release_first.setdefault(fruit, frame)
            release_last.add(fruit)
        else:
            check(fruit not in release_last, f'{path.name}: release cleared {offset}')
        gap = finite_bench_gap(values['x_m'], values['y_m'], values['z_m'], radius)
        minimum_gap = min(minimum_gap, gap)
        maximum_gap_error = max(maximum_gap_error, abs(gap - values['static_clearance_m']))
        parsed[frame, fruit] = values
    check(maximum_gap_error <= 1e-9, f'{path.name}: exported static gap mismatch')
    final = [parsed[FRAMES, fruit] for fruit in range(12)]
    grounded_released = []
    lower_floor_released = []
    for fruit in sorted(release_last):
        row = final[fruit]
        speed = math.sqrt(sum(row[key] ** 2 for key in ('vx_m_s', 'vy_m_s', 'vz_m_s')))
        if abs(row['static_clearance_m']) <= 2e-6 and speed <= 1e-3:
            grounded_released.append(fruit)
        if abs(row['z_m'] + .75 - row['radius_m']) <= 2e-6 and speed <= 1e-3:
            lower_floor_released.append(fruit)
    return {'sha256': hashlib.sha256(payload).hexdigest(), 'rows': len(rows),
            'minimum_independent_static_gap_m': minimum_gap,
            'maximum_exported_gap_discrepancy_m': maximum_gap_error,
            'released_fruits': sorted(release_last), 'first_release_frame': release_first,
            'grounded_released_fruits': grounded_released,
            'lower_room_floor_released_fruits': lower_floor_released}, parsed, payload


def snapshot(path, frame, states):
    payload = path.read_bytes()
    lines = payload.decode().splitlines()
    check(lines[:3] == ['# Numi Solver explicit-yarn Metal cloth bag',
          '# static_bench min -0.75 -0.5 -0.08 max 0.75 0.5 0 floor -0.75',
          '# vertices 1465 render_triangles 2880'], f'{path.name}: header')
    vertices = 0
    faces = 0
    balls = {}
    minimum_vertex_z = math.inf
    for line in lines[3:]:
        if line.startswith('v '):
            values = list(map(float, line.split()[1:]))
            check(len(values) == 3 and all(map(math.isfinite, values)), f'{path.name}: vertex')
            vertices += 1
            minimum_vertex_z = min(minimum_vertex_z, values[2])
        elif line.startswith('f '):
            indices = [int(value) for value in line.split()[1:]]
            check(len(indices) == 3 and all(1 <= value <= 1465 for value in indices), f'{path.name}: face')
            faces += 1
        elif line.startswith('# ball '):
            words = line.split()
            fruit = int(words[2])
            check(fruit not in balls and words[3] == 'center' and words[7] == 'radius', f'{path.name}: fruit')
            center = list(map(float, words[4:7]))
            radius = float(words[8])
            vindex = words.index('linear_velocity')
            velocity = list(map(float, words[vindex + 1:vindex + 4]))
            check(len(velocity) == 3 and all(map(math.isfinite, center + [radius] + velocity)), f'{path.name}: fruit values')
            row = states[frame, fruit]
            expected = [row[key] for key in ('x_m', 'y_m', 'z_m', 'radius_m', 'vx_m_s', 'vy_m_s', 'vz_m_s')]
            check(max(abs(a - b) for a, b in zip(center + [radius] + velocity, expected)) <= 1e-7,
                  f'{path.name}: fruit/CSV mismatch')
            balls[fruit] = True
    check(vertices == 1465 and faces == 2880 and set(balls) == set(range(12)), f'{path.name}: mesh counts')
    return {'sha256': hashlib.sha256(payload).hexdigest(), 'bytes': len(payload),
            'minimum_vertex_z_m': minimum_vertex_z}, payload


def main():
    launch_path = pathlib.Path(str(PREFIX) + '.launch.json')
    exit_path = pathlib.Path(str(PREFIX) + '.exit')
    check(launch_path.is_file() and exit_path.is_file(), 'terminal launch and actual exit marker required')
    launch = json.loads(launch_path.read_text())
    check(launch.get('status') == 'COMPLETE' and 'exit_code' in launch and 'elapsed_seconds' in launch,
          'complete launch record required')
    pid = int(launch['pid'])
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        pass
    else:
        raise ValueError(f'PID {pid} still exists')
    actual_exit = int(exit_path.read_text().strip())
    check(actual_exit == launch['exit_code'], 'launch/marker exit mismatch')
    command = launch['command']
    expected_command = [str(ROOT / 'build/native-finite-repaired-launch/build/numi-solver-cloth-metal'),
                        '--metallib', str(ROOT / 'build/native-finite-repaired-launch/build/shaders/NumiTemporalCone.metallib'),
                        '--finite-bench', '--pickup-prefix', str(PREFIX), '--pickup-steps', '480',
                        '--pickup-dump-every', '10', '--iterations', '32', '--strain-sweeps', '3',
                        '--replays', '2']
    check(command == expected_command, 'launch command mismatch')
    check(launch['requested_frames_per_replay'] == 480 and launch['requested_replays'] == 2 and
          launch['requested_simulated_seconds'] == 4 and launch['substeps_per_frame'] == 48 and
          launch['iterations'] == 32 and launch['strain_sweeps'] == 3 and
          launch['static_box_min_m'] == [-.75, -.5, -.08] and
          launch['static_box_max_m'] == [.75, .5, 0] and launch['room_floor_height_m'] == -.75,
          'launch configuration mismatch')
    manifest_path = pathlib.Path(launch['frozen_manifest'])
    check(sha(manifest_path) == launch['manifest_sha256'], 'frozen manifest hash mismatch')
    manifest = json.loads(manifest_path.read_text())
    check(manifest['source_commit'] == launch['source_commit'], 'source commit mismatch')
    for name, record in manifest['files'].items():
        path = manifest_path.parent / name
        check(path.stat().st_size == record['bytes'] and sha(path) == record['sha256'], f'frozen source changed: {name}')
    summaries = []
    states = {}
    payloads = []
    for replay in (1, 2):
        path = pathlib.Path(str(PREFIX) + f'-r{replay}-fruits.csv')
        summary, entries, payload = trace(path, replay)
        summaries.append(summary)
        states[replay] = entries
        payloads.append(payload)
    check(states[1] == states[2], 'exported fruit frame replay differs')
    combined = pathlib.Path(str(PREFIX) + '-fruits.csv')
    header = (HEADER + '\n').encode()
    check(combined.read_bytes() == payloads[0] + payloads[1][len(header):], 'combined trace differs from per-replay traces')
    sampled = []
    for frame in range(0, FRAMES + 1, 10):
        first_path = pathlib.Path(str(PREFIX) + f'-{frame}.obj')
        second_path = pathlib.Path(str(PREFIX) + f'-r2-{frame}.obj')
        one, raw1 = snapshot(first_path, frame, states[1])
        two, raw2 = snapshot(second_path, frame, states[2])
        check(raw1 == raw2, f'OBJ replay differs at frame {frame}')
        sampled.append({'frame': frame, **one})
    log_path = pathlib.Path(str(PREFIX) + '.log')
    log = log_path.read_text()
    progress = {}
    for match in re.finditer(r'^pickup_progress scenario=pickup replay=(\d) step=(\d+)/480 released_mask=(\d+) max_local_node_overlap_m=([^ ]+) failure_flags=(\d+) gpu_seconds=([^\n]+)$', log, re.M):
        replay, frame, mask, overlap, failure, seconds = match.groups()
        replay, frame, mask = int(replay), int(frame), int(mask)
        check(replay in (1, 2) and frame in range(10, 481, 10), 'progress identity')
        check(int(failure) == 0 and float(overlap) <= 2e-6 and float(seconds) > 0, 'progress gate')
        check(mask == sum((1 << fruit) for fruit in range(12) if states[replay][frame, fruit]['released']),
              'progress release mask differs from CSV')
        check((replay, frame) not in progress, 'duplicate progress')
        progress[replay, frame] = {'released_mask': mask, 'max_local_node_overlap_m': float(overlap),
                                   'gpu_seconds': float(seconds)}
    check(len(progress) == 96, 'incomplete per-replay progress lines')
    pickup_lines = re.findall(r'^(pickup_requested=[^\n]+)$', log, re.M)
    result_lines = re.findall(r'^result=(PASS|FAIL)$', log, re.M)
    check(len(pickup_lines) == len(result_lines) == 1, 'missing terminal native summary')
    fields = dict(token.split('=', 1) for token in pickup_lines[0].split())
    check(fields['complete'] == 'true' and fields['first_captured_frames'] == '480' and
          fields['second_captured_frames'] == '480' and fields['steps'] == '480', 'native incomplete')
    check(fields['first_failure_flags'] == fields['second_failure_flags'] == '0', 'native failure flags')
    check(fields['replay_exact'] == 'true', 'native whole-state replay failed')
    check(int(fields['released_mask']) == progress[1, 480]['released_mask'], 'summary release mask')
    check(int(fields['released_count']) == len(summaries[0]['released_fruits']), 'summary release count')
    check(int(fields['grounded_released_count']) == len(summaries[0]['grounded_released_fruits']),
          'summary grounded count')
    check(abs(float(fields['max_local_node_overlap_m']) - max(progress[r, 480]['max_local_node_overlap_m'] for r in (1, 2))) <= 1e-12,
          'summary local overlap')
    check((result_lines[0] == 'PASS') == (actual_exit == 0), 'result/actual exit mismatch')
    result = {'schema': 'numi.native-finite-repaired-pickup-independent-cpu-audit.v1',
              'actual_program_exit': actual_exit, 'native_result': result_lines[0],
              'launch_sha256': sha(launch_path), 'manifest_sha256': sha(manifest_path),
              'frozen_binary_sha256': manifest['files']['build/numi-solver-cloth-metal']['sha256'],
              'frozen_metallib_sha256': manifest['files']['build/shaders/NumiTemporalCone.metallib']['sha256'],
              'log_sha256': sha(log_path), 'combined_csv_sha256': sha(combined),
              'native_pickup_summary': fields, 'per_replay_csv': summaries,
              'all_481_exported_fruit_frames_exact': True, 'all_49_sampled_obj_pairs_exact': True,
              'snapshots': sampled, 'progress_lines_verified': len(progress),
              'minimum_sampled_cloth_vertex_z_m': min(row['minimum_vertex_z_m'] for row in sampled),
              'scope': ['Native internal 480-frame hash equality is reported by the frozen binary; hash lists are not exported for independent recomputation.',
                        'Every exported fruit frame and 10-frame-cadence OBJ pair is independently compared.',
                        'Independent fruit-to-finite-box/room-floor static gaps and CSV consistency are checked.',
                        'OBJ snapshots do not expose all yarn/contact pairs, per-substep contact, strain, or a physical energy/work ledger.',
                        'Terminal native overlap/strain fields are reported, not promoted to independently established whole-trajectory contact or energy closure.']}
    OUTPUT.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: result[key] for key in ('actual_program_exit', 'native_result',
          'all_481_exported_fruit_frames_exact', 'all_49_sampled_obj_pairs_exact',
          'progress_lines_verified', 'minimum_sampled_cloth_vertex_z_m')}, indent=2))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError, IndexError) as error:
        print(f'audit not completed: {error}', file=sys.stderr)
        raise SystemExit(2)
