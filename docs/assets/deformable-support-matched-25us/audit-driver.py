"""Read-only terminal L3 geometry/energy audit followed by matched L3/L4 spatial comparison."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import audit_deformable_mesh_energy as energy
import audit_deformable_mesh_refinement as refinement


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def manifest_ok():
    records = []
    for line in (ROOT / 'build/source-binary.sha256').read_text().splitlines():
        if not line.strip():
            continue
        digest, name = line.split(None, 1)
        path = ROOT / name.lstrip('*')
        if sha(path) != digest:
            raise RuntimeError('Changed immutable source/binary: ' + name)
        records.append({'path': name, 'sha256': digest})
    return records


def main():
    if (ROOT / 'build/mesh4-level3-dt25us-match.exit').read_text().strip() != '0':
        raise RuntimeError('L3 has no actual exit zero')
    for pid in (87159, 86349):
        if subprocess.run(['ps', '-p', str(pid), '-o', 'command='], capture_output=True).returncode == 0:
            raise RuntimeError('L3 owner/queue still present')
    original_sources = manifest_ok()
    inputs = {str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT / 'build').glob('mesh4-level3-dt25us-match*')) if p.is_file()}
    geometry_function = energy.audit_geometry
    observations = []
    begin = time.time()
    for label, suffix, steps in (('primary', '', 20000), ('half', '-half', 40000)):
        prefix = ROOT / ('build/mesh4-level3-dt25us-match' + suffix)
        current_geometry = []

        def geometry_once(path):
            report = geometry_function(path)
            current_geometry.append(report)
            return report

        energy.audit_geometry = geometry_once
        (HERE / 'status.json').write_text(json.dumps({'status': 'AUDITING', 'case': label,
            'pid': os.getpid(), 'completed_cases': len(observations)}) + '\n')
        report = energy.audit(prefix, steps)
        geometry = current_geometry[0]
        if len(current_geometry) != 1:
            raise RuntimeError('Unexpected duplicate geometry audit')
        (HERE / (label + '-geometry.json')).write_text(json.dumps(geometry, indent=2) + '\n')
        (HERE / (label + '-energy.json')).write_text(json.dumps(report, indent=2) + '\n')
        observations.append({'case': label, 'captured_states': geometry['captured_frames'],
            'geometry_pass': True, 'energy_accounting_result': report['result'],
            'one_percent_numerical_budget_pass': report['one_percent_numerical_energy_budget_pass'],
            'numerical_bound_fraction': report['energy_error_bound_fraction_of_initial'],
            'maximum_energy_accounting_residual_J': report['maximum_energy_accounting_residual_J']})
        print(json.dumps(observations[-1]), flush=True)
    for path, digest in inputs.items():
        if sha(ROOT / path) != digest:
            raise RuntimeError('Audited immutable input changed: ' + path)
    if manifest_ok() != original_sources:
        raise RuntimeError('Source/binary manifest changed')
    spatial = refinement.compare([ROOT/'build/mesh4-level3-dt25us-match', ROOT/'build/mesh4-level4-dt25us'], .001)
    (HERE/'spatial.json').write_text(json.dumps(spatial, indent=2) + '\n')
    print(json.dumps({'case':'matched-spatial','within_1mm_target':spatial['finest_pair_within_spatial_target'],
        'maximum_matched_node_difference_m':spatial['comparison'][0]['maximum_matched_node_difference_m']}), flush=True)
    evidence = {'schema': 'numi.deformable.L3.matched-native-independent-audit.v1',
        'actual_solver_exit': 0, 'retained_input_hashes': inputs,
        'immutable_source_binary_entries': original_sources,
        'source_binary_manifest_sha256': sha(ROOT / 'build/source-binary.sha256'),
        'audit_source_sha256': {p.name: sha(p) for p in HERE.glob('*.py')},
        'observations': observations, 'spatial_result': spatial, 'runtime_s': time.time() - begin,
        'scope': 'Complete101capturedstatespercase with owning native source/actual exit. Native replay/all-step claims remain source telemetry. Spatial comparison is a distinct measured result; material calibration, reciprocal yarn contact and physical work closure remain open.',
        'new_GPU_work': False}
    (HERE / 'evidence.json').write_text(json.dumps(evidence, indent=2) + '\n')
    code = 0 if all(o['energy_accounting_result'] == 'PASS' for o in observations) else 1
    (HERE / 'status.json').write_text(json.dumps({'status': 'TERMINAL', 'actual_audit_exit': code,
        'completed_cases': len(observations), 'spatial_within_1mm_target': spatial['finest_pair_within_spatial_target']}) + '\n')
    return code


if __name__ == '__main__':
    try:
        code = main()
    except Exception as error:
        print('FAIL', repr(error), flush=True)
        (HERE / 'status.json').write_text(json.dumps({'status': 'FAILED', 'error': repr(error), 'pid': os.getpid()}) + '\n')
        code = 1
    raise SystemExit(code)
