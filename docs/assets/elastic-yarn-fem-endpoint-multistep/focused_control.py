#!/usr/bin/env python3
"""Source-bound step-13 old-centroid rejection and endpoint continuation."""
import copy
import gzip
import importlib.util
import json
import pathlib
import struct
import sys
import time

P = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('endpoint_control', P / 'endpoint_step.py')
E = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E)
B = E.B
OLD = B.ROOT / 'build/elastic-yarn-fem-multistep-short/candidate-20us'
expected_old_state = '117f4863ba01b8626b518f471b07c11a156390405bf5e04b435823f145e2aec9'


def main():
    start = time.time()
    old_receipt = json.loads((OLD / 'receipt.json').read_text())
    assert old_receipt['result_sha256'] == B.sha(OLD / 'result.json')
    old_result = json.loads((OLD / 'result.json').read_text())
    assert old_result['first_unsupported_frontier']['reason'] == 'contact-manifold/gap'
    assert old_result['first_unsupported_frontier']['attempted_step'] == 13
    assert old_result['terminal_checkpoint']['complete_state_sha256'] == expected_old_state
    dt = old_result['dt_s']
    meta = json.loads(gzip.decompress((OLD / 'terminal.metadata.json.gz').read_bytes()))
    records = list(struct.iter_unpack('<6d', (OLD / 'terminal.state.f64x6.bin').read_bytes()))
    assert len(records) == 2059
    state = copy.deepcopy(meta)
    state['x'] = [list(q[:3]) for q in records]
    state['v'] = [list(q[3:]) for q in records]
    assert B.state_sha(state) == expected_old_state
    assert state['steps'] == 12 and abs(state['elapsed'] - 12 * dt) < 1e-18
    old = copy.deepcopy(state)
    old_out = E.N.advance(E.N.Model(), old, dt)
    assert not old_out['accepted'] and old_out['reason'] == 'contact-manifold/gap'
    assert old_out['rollback_exact'] and B.state_sha(old) == expected_old_state
    new_model = E.Model()
    new = copy.deepcopy(state)
    first = E.advance(new_model, new, dt)
    assert first['accepted']
    replay = copy.deepcopy(state)
    second = E.advance(E.Model(), replay, dt)
    assert second['accepted'] and B.state_sha(replay) == B.state_sha(new)
    gap = first['ledger']['final_global_velocity_scan']['minimum']['gap_m']
    rate = first['ledger']['final_global_velocity_scan']['minimum']['closing_m_s']
    assert gap >= -B.GAP_GATE and rate >= -512 * B.EPS
    controls = []
    for fault in ('invalid-J', 'invalid-area'):
        victim = copy.deepcopy(state)
        rejected = E.advance(E.Model(), victim, dt, fault=fault)
        controls.append({'fault': fault, 'accepted': rejected['accepted'],
                         'reason': rejected.get('reason'),
                         'rollback_exact': rejected.get('rollback_exact'),
                         'complete_state_unchanged': B.state_sha(victim) == expected_old_state})
    assert all(not q['accepted'] and q['rollback_exact'] and q['complete_state_unchanged'] for q in controls)
    result = {'schema': 'numi.elastic-yarn.endpoint-step13-focused-control.v1',
              'passed': True,
              'source_sha256': {str(q): B.sha(q) for q in
                                (P / 'endpoint_step.py', P / 'step.py', P / 'centroid_step.py', pathlib.Path(__file__))},
              'old_trajectory_result_sha256': B.sha(OLD / 'result.json'),
              'old_terminal_buffer_sha256': B.sha(OLD / 'terminal.state.f64x6.bin'),
              'old_terminal_metadata_sha256': B.sha(OLD / 'terminal.metadata.json.gz'),
              'pre_state_sha256': expected_old_state,
              'dt_s': dt,
              'old_rejected_reason': old_out['reason'],
              'old_rejected_detail': old_out['detail'],
              'old_rollback_exact': old_out['rollback_exact'],
              'endpoint_accepted': first['accepted'], 'endpoint_exact_replay': True,
              'endpoint_state_sha256': B.state_sha(new),
              'endpoint_final_minJ': first['final_eval']['minJ'],
              'endpoint_final_global_gap_m': gap,
              'endpoint_final_global_normal_rate_m_s': rate,
              'endpoint_final_closest_u': first['ledger']['final_global_velocity_scan']['minimum']['u'],
              'endpoint_final_closest_face': first['ledger']['final_global_velocity_scan']['minimum']['face'],
              'rollback_negative_controls': controls,
              'runtime_s': time.time() - start,
              'scope': 'One private CPU continuation from a reproduced 12-step state; numerical FP64 witness, no CCD or native trajectory claim.'}
    target = P / 'focused-control.json'
    target.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: result[k] for k in ('passed', 'old_rejected_reason',
                                          'endpoint_accepted', 'endpoint_exact_replay',
                                          'endpoint_final_global_gap_m', 'endpoint_final_minJ',
                                          'rollback_negative_controls', 'runtime_s')}, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
