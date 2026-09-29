#!/usr/bin/env python3
"""Audit every stored accepted contact feature in a complete checkpoint."""
import argparse
import gzip
import hashlib
import importlib.util
import json
import math
import pathlib
import struct
import sys

P = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('endpoint_geometry_audit', P / 'endpoint_step.py')
E = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E)
B = E.B


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=pathlib.Path, required=True)
    parser.add_argument('--output', type=pathlib.Path, required=True)
    args = parser.parse_args()
    directory = args.input.resolve()
    result = json.loads((directory / 'result.json').read_text())
    binding = result['terminal_checkpoint']
    metadata_file = directory / binding['metadata_file']
    raw_file = directory / binding['position_file']
    assert B.sha(metadata_file) == binding['metadata_sha256']
    assert B.sha(raw_file) == binding['position_sha256']
    state = json.loads(gzip.decompress(metadata_file.read_bytes()))
    raw = list(struct.iter_unpack('<6d', raw_file.read_bytes()))
    assert len(raw) == 2059
    state['x'] = [list(q[:3]) for q in raw]
    state['v'] = [list(q[3:]) for q in raw]
    assert B.state_sha(state) == binding['complete_state_sha256']
    ledgers = state['ledger']
    records = result.get('primary_records', result.get('additional_records'))
    offset = len(ledgers) - len(records)
    assert offset >= 0 and state['steps'] == len(ledgers)
    model = E.Model()
    selected = model.row[2:]
    rows = []
    for local_i, record in enumerate(records):
        index = offset + local_i
        ledger = ledgers[index]
        geometry = ledger['final_geometry']
        closest = geometry['oracle']
        scan = ledger['final_global_velocity_scan']
        minimum = scan['minimum']
        endpoint = geometry['endpoint_rows']
        assert len(endpoint) == 2
        assert ledger['final_areas']['faces'] == scan['faces_checked'] == 1280
        assert minimum['nodes'] == selected
        assert record['minimum_gap_m'] == minimum['gap_m']
        assert record['minimum_closing_m_s'] == minimum['closing_m_s']
        assert record['minJ'] == ledger['end']['minJ']
        assert ledger['end']['minJ'] > B.J_GATE
        assert closest['valid'] and 0 <= closest['u'] <= 1
        assert min(closest['bary']) > 64 * B.EPS
        assert all(min(q['bary']) > 64 * B.EPS for q in endpoint)
        assert all(-B.GAP_GATE <= q['gap'] <= B.GAP_GATE for q in endpoint)
        assert -B.GAP_GATE <= geometry['gap'] <= B.GAP_GATE
        assert -B.GAP_GATE <= minimum['gap_m'] <= B.GAP_GATE
        assert B.finite([geometry['distance'], minimum['distance'], minimum['closing_m_s']])
        scale = max(1., *(B.norm(state['v'][i]) for i in model.row))
        # The final-state owner scale is a conservative reporting check; the
        # frozen solver enforced the precise state-specific scale at each step.
        assert minimum['closing_m_s'] >= -512 * B.EPS * scale
        assert all(q['nodes'] == selected for q in scan['contacts_within_2um'])
        rows.append({'absolute_step': index + 1, 'elapsed_s': record['elapsed_s'],
                     'global_face': minimum['face'],
                     'global_closest_u': minimum['u'],
                     'global_closest_bary': minimum['bary'],
                     'selected_closest_u': closest['u'],
                     'selected_closest_bary': closest['bary'],
                     'endpoint_bary': [q['bary'] for q in endpoint],
                     'endpoint_gaps_m': [q['gap'] for q in endpoint],
                     'global_gap_m': minimum['gap_m'],
                     'global_normal_rate_m_s': minimum['closing_m_s'],
                     'minJ': ledger['end']['minJ'],
                     'contact_faces_within_2um': len(scan['contacts_within_2um'])})
    first_endpoint = next((q['absolute_step'] for q in rows
                           if q['global_closest_u'] in (0., 1.)), None)
    audit = {'schema': 'numi.elastic-yarn.full-checkpoint-feature-history.v1',
             'passed': True, 'result_sha256': B.sha(directory / 'result.json'),
             'complete_checkpoint_sha256': binding['complete_state_sha256'],
             'checkpoint_metadata_sha256': B.sha(metadata_file),
             'checkpoint_owner_buffer_sha256': B.sha(raw_file),
             'auditor_source_sha256': B.sha(pathlib.Path(__file__)),
             'endpoint_source_sha256': B.sha(P / 'endpoint_step.py'),
             'absolute_step_start': offset + 1,
             'absolute_step_end': len(ledgers),
             'accepted_steps_audited': len(rows),
             'first_endpoint_global_feature_step': first_endpoint,
             'minimum_endpoint_barycentric_weight': min(q for row in rows for beta in row['endpoint_bary'] for q in beta),
             'minimum_selected_closest_barycentric_weight': min(q for row in rows for q in row['selected_closest_bary']),
             'minimum_global_closest_barycentric_weight': min(q for row in rows for q in row['global_closest_bary']),
             'minimum_global_gap_m': min(row['global_gap_m'] for row in rows),
             'maximum_abs_global_normal_rate_m_s': max(abs(row['global_normal_rate_m_s']) for row in rows),
             'all_selected_face_interior_and_two_endpoint_rows_admitted': True,
             'all_global_contact_faces_selected': True,
             'rows': rows}
    args.output.write_text(json.dumps(audit, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: audit[k] for k in ('passed', 'accepted_steps_audited',
                                         'first_endpoint_global_feature_step',
                                         'minimum_endpoint_barycentric_weight',
                                         'minimum_global_gap_m')}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
