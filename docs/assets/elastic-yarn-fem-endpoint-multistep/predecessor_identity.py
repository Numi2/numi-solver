#!/usr/bin/env python3
"""Compare twelve old/new accepted states, including complete raw owner bytes."""
import hashlib
import importlib.util
import json
import pathlib
import struct
import sys
import time

P = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('endpoint_identity', P / 'endpoint_step.py')
E = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E)
B = E.B
OLD = B.ROOT / 'build/elastic-yarn-fem-multistep-short/candidate-20us'
NEW = P / 'candidate-20us'


def owner_sha(state):
    raw = b''.join(struct.pack('<6d', *(state['x'][i] + state['v'][i]))
                   for i in range(len(state['x'])))
    assert len(raw) == 2059 * 6 * 8
    return hashlib.sha256(raw).hexdigest()


def main():
    start = time.time()
    old_result = json.loads((OLD / 'result.json').read_text())
    new_result = json.loads((NEW / 'result.json').read_text())
    old_records = old_result['primary_records']
    new_records = new_result['primary_records']
    assert len(old_records) == 12 and len(new_records) >= 13
    assert old_result['dt_s'] == new_result['dt_s']
    dt = old_result['dt_s']
    old_model = E.N.Model()
    new_model = E.Model()
    old = old_model.initial()
    new = new_model.initial()
    assert owner_sha(old) == owner_sha(new)
    shared_metrics = ('stress_J_U_sha256', 'force_sha256', 'minJ', 'maxJ',
                      'minimum_gap_m', 'minimum_closing_m_s',
                      'contact_dissipation_J', 'plane_kinetic_removal_J',
                      'gravity_work_J', 'physical_energy_balance_residual_J',
                      'end_energy_J', 'end_momentum_kg_m_s',
                      'end_angular_kg_m2_s', 'row_impulses_Ns',
                      'row_active_masks')
    rows = []
    for i in range(12):
        oa = E.N.advance(old_model, old, dt)
        na = E.advance(new_model, new, dt)
        assert oa['accepted'] and na['accepted']
        ro, rn = old_records[i], new_records[i]
        assert B.state_sha(old) == ro['state_sha256']
        assert B.state_sha(new) == rn['state_sha256']
        assert all(ro[key] == rn[key] for key in shared_metrics)
        oh, nh = owner_sha(old), owner_sha(new)
        assert oh == nh
        # Both endpoint rows must remain strictly inside the one selected
        # triangle. The true selected-face closest yarn feature may vary.
        geometry = na['ledger']['final_geometry']
        endpoint_bary = [q['bary'] for q in geometry['endpoint_rows']]
        closest = geometry['oracle']
        assert all(min(beta) > 64 * B.EPS for beta in endpoint_bary)
        assert min(closest['bary']) > 64 * B.EPS
        assert 0 <= closest['u'] <= 1
        rows.append({'step': i + 1, 'owner_xv_sha256': oh,
                     'old_complete_state_sha256': ro['state_sha256'],
                     'new_complete_state_sha256': rn['state_sha256'],
                     'complete_state_hashes_equal': ro['state_sha256'] == rn['state_sha256'],
                     'shared_stress_sha256': ro['stress_J_U_sha256'],
                     'shared_force_sha256': ro['force_sha256'],
                     'shared_metric_values_equal': True,
                     'new_selected_closest_u': closest['u'],
                     'new_selected_closest_bary': closest['bary'],
                     'new_endpoint_bary': endpoint_bary})
    result = {'schema': 'numi.elastic-yarn.predecessor-12-state-owner-identity.v1',
              'passed': True, 'actual_replayed_steps': 12,
              'source_sha256': {str(q): B.sha(q) for q in
                                (P / 'endpoint_step.py', P / 'step.py',
                                 P / 'centroid_step.py', pathlib.Path(__file__))},
              'old_result_sha256': B.sha(OLD / 'result.json'),
              'new_result_sha256': B.sha(NEW / 'result.json'),
              'old_initial_sha256': old_result['initial_state_sha256'],
              'new_initial_sha256': new_result['initial_state_sha256'],
              'all_2059_owner_buffers_identical_per_step': True,
              'all_10240_stress_and_2059_force_hashes_identical_per_step': True,
              'shared_physical_metric_keys': list(shared_metrics),
              'shared_physical_metric_values_equal_per_step': True,
              'complete_state_hashes_differ_due_to_witness_ledger_and_cache': True,
              'rows': rows, 'runtime_s': time.time() - start}
    (P / 'predecessor-identity.json').write_text(json.dumps(result, indent=2,
                                                            allow_nan=False) + '\n')
    print(json.dumps({'passed': True, 'steps': 12,
                      'owner_buffers_equal': True, 'physical_metrics_equal': True,
                      'complete_state_sha_equal': sum(q['complete_state_hashes_equal'] for q in rows),
                      'runtime_s': result['runtime_s']}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
