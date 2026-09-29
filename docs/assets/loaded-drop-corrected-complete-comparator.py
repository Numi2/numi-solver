"""Compare complete retained CPU48/CPU96 drops without accepting failed cases."""
import csv, hashlib, io, json, math, sys
from pathlib import Path

root = Path('/Users/home/numi-solver')
base = root / 'build'
audit_root = base / 'loaded-drop-investigation-terminal-audit'
sys.path.insert(0, str(audit_root))
from audit_cloth_snapshot import read_snapshot, topology

def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
manifest_path = base / 'loaded-drop-investigation-corrected-launch/manifest.json'
manifest = json.loads(manifest_path.read_text())
inputs = {str(manifest_path): digest(manifest_path)}
for rel, expected in manifest['sha256'].items():
    p = root / rel
    assert digest(p) == expected, str(p)
    inputs[str(p)] = expected
status_path = audit_root / 'status.json'
status = json.loads(status_path.read_text())
assert status['status'] == 'TERMINAL'
inputs[str(status_path)] = digest(status_path)
commands, audits, geometry, traces = {}, {}, {}, {}
for count in (48, 96):
    count_key = str(count)
    p = base / f'loaded-drop-investigation-corrected-{count}'
    launch_path = Path(str(p) + '.launch.json')
    launch = json.loads(launch_path.read_text())
    commands[count_key] = list(launch['command'])
    for option in ('--dump-frames','--dump-obj','--fruit-trace','--contact-peak-prefix'):
        idx = commands[count_key].index(option) + 1
        commands[count_key][idx] = '<unique-output>'
    commands[count_key][commands[count_key].index('--substeps') + 1] = '<substeps>'
    assert int(Path(str(p) + '.exit').read_text().strip()) == launch['actual_exit_code']
    assert status['cases'][count_key]['actual_solver_exit'] == launch['actual_exit_code']
    case = status['cases'][count_key]
    assert case['status'] == 'TERMINAL_AUDITED'
    assert case['all_seven_frozen_physics_hashes_verified'] and case['immutable_inputs_verified_after_audits']
    for name, expected in case['immutable_input_sha256'].items():
        path = Path(name); assert digest(path) == expected, name; inputs[name] = expected
    audit_path = audit_root / f'{count}-loaded-drop-audit.json'
    geo_path = audit_root / f'{count}-all-saved-geometry.json'
    audits[count_key] = json.loads(audit_path.read_text())
    geometry[count_key] = json.loads(geo_path.read_text())
    for path in (launch_path, audit_path, geo_path): inputs[str(path)] = digest(path)
    payload = Path(str(p) + '-fruits.csv').read_bytes()
    states = {}
    for row in csv.DictReader(io.StringIO(payload.decode())):
        identity = tuple(int(row[k]) for k in ('replay','frame','fruit'))
        assert identity not in states
        states[identity] = row
    assert set(states) == {(replay,frame,fruit) for replay in (1,2) for frame in range(721) for fruit in range(12)}
    assert all({k:v for k,v in states[1,f,i].items() if k != 'replay'} ==
               {k:v for k,v in states[2,f,i].items() if k != 'replay'} for f in range(721) for i in range(12))
    traces[count_key] = states
assert commands['48'] == commands['96']
faces = topology()[1]
masses = [0.0001 if 26 * 48 <= i < 28 * 48 else .00005 for i in range(1465)]
rows = []; worst = {'metres': -1}; rms_peak = {'metres': -1}; com_peak = {'metres': -1}
for frame in range(0,721,10):
    vertices = {}
    for count in ('48','96'):
        p = base / f'loaded-drop-investigation-corrected-{count}-{frame}.obj'
        payload, vertices[count], _ = read_snapshot(p,faces)
        assert hashlib.sha256(payload).hexdigest() == inputs[str(p)]
    diffs = [math.dist(a,b) for a,b in zip(vertices['48'],vertices['96'])]
    node = max(range(1465), key=diffs.__getitem__)
    rms = math.sqrt(sum(m*d*d for m,d in zip(masses,diffs))/sum(masses))
    coms = {count: [sum(m*p[j] for m,p in zip(masses,points))/sum(masses) for j in range(3)] for count,points in vertices.items()}
    com_diff = math.dist(coms['48'],coms['96'])
    row = {'frame':frame,'time_s':frame/120,'maximum_corresponding_cloth_position_difference_m':diffs[node],
           'cloth_node':node,'mass_weighted_rms_position_difference_m':rms,'cloth_mass_center_difference_m':com_diff}
    rows.append(row)
    if diffs[node] > worst['metres']: worst = {'metres':diffs[node],'frame':frame,'node':node}
    if rms > rms_peak['metres']: rms_peak = {'metres':rms,'frame':frame}
    if com_diff > com_peak['metres']: com_peak = {'metres':com_diff,'frame':frame}
fruit_pos = {'metres':-1}; fruit_vel = {'metres_per_second':-1}; release_difference = None
for frame in range(721):
    for fruit in range(12):
        a,b = traces['48'][1,frame,fruit],traces['96'][1,frame,fruit]
        dp = math.dist([float(a[f'{k}_m']) for k in 'xyz'],[float(b[f'{k}_m']) for k in 'xyz'])
        dv = math.dist([float(a[f'v{k}_m_s']) for k in 'xyz'],[float(b[f'v{k}_m_s']) for k in 'xyz'])
        if dp > fruit_pos['metres']: fruit_pos = {'metres':dp,'frame':frame,'fruit':fruit}
        if dv > fruit_vel['metres_per_second']: fruit_vel = {'metres_per_second':dv,'frame':frame,'fruit':fruit}
        if release_difference is None and a['released'] != b['released']: release_difference = frame
summary = {}
for count in ('48','96'):
    a = audits[count]; g = geometry[count]
    summary[count] = {'actual_solver_exit':a['actual_exit_code'],'actual_solver_result':a['actual_solver_result'],
        'qualified_authored_cpu_drop':a['qualified_loaded_cloth_drop'],
        'solver_reports_both_ordered_721_frame_replays_match':a['solver_reports_both_ordered_721_frame_replays_match'],
        'serialized_fruit_replay_exact':a['fruit_trace']['serialized_fruit_replay_exact'],
        'complete_saved_snapshot_count':g['all_complete_saved_snapshot_count'],
        'all_saved_geometry_pass':g['all_snapshots_within_contact_tolerance'],
        'cloth_center_descent_m':a['cloth_center_descent_after_first_post_release_snapshot_m'],
        'first_saved_cloth_floor_contact_time_s':a['first_saved_post_release_lower_floor_contact_time_s'],
        'final_floor_contact_node_count':a['final_lower_floor_contact_node_count'],
        'released_fruits':a['fruit_trace']['replays'][0]['released_fruits'],
        'released_fruits_landed_on_floor':a['fruit_trace']['replays'][0]['released_fruits_landed_on_floor']}
result = {'schema':'numi.loaded-drop.matched-complete-CPU-timestep-sensitivity.v1',
    'actual_terminal_cases':summary,'all_seven_frozen_physics_hashes_verified':True,
    'normalized_launch_commands_match':True,'only_command_differences':'Substeps48/96 and four unique output paths.',
    'frames_per_fruit_replay':721,'saved_cloth_comparison_frames':73,'duration_s':6,
    'first_release_bit_difference_frame':release_difference,'maximum_corresponding_fruit_position_difference':fruit_pos,
    'maximum_corresponding_fruit_velocity_difference':fruit_vel,'maximum_corresponding_saved_cloth_position_difference':worst,
    'maximum_mass_weighted_saved_cloth_rms_difference':rms_peak,'maximum_saved_cloth_mass_center_difference':com_peak,
    'per_saved_frame_cloth_comparison':rows,'release_outcome_timestep_convergence':'FAIL; complete released sets differ.',
    'both_authored_cases_qualified':False,'source_and_terminal_inputs_sha256':inputs,
    'comparator_sha256':digest(Path(__file__)),
    'boundary':'Complete terminal replay and saved-state comparison only. CPU96 is a retained failed case, not an accepted finer solution. Cloth positions are serialized first-replay captures every10frames; intervening cloth motion is not reconstructed. Fruit differences are over every frame after exact full replay comparison. No calibrated material, energy/work closure, native or asymptotic convergence claim.'}
out = Path(__file__).with_name('comparison.json')
out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('source_and_terminal_inputs_sha256','per_saved_frame_cloth_comparison')},indent=2))
raise SystemExit(1 if release_difference is not None else 0)
