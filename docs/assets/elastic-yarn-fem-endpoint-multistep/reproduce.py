#!/usr/bin/env python3
"""Portable fresh-root replay of the 640 us full-FEM/endpoint-yarn CPU study."""
import gzip
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time

HERE = pathlib.Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False,
                      separators=(',', ':')).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def physical_result(result):
    return {key: result[key] for key in ('primary_accepted_steps',
            'first_unsupported_frontier', 'right_censored_at_max_steps',
            'exact_replay', 'full_body_summary', 'primary_records',
            'terminal_checkpoint', 'matched_interval')}


def continuation_result(result):
    return {key: result[key] for key in ('dt_s', 'bit_identical_candidate_dt',
            'preceding_32step_terminal_state_sha256',
            'additional_accepted_steps', 'total_accepted_steps',
            'first_unsupported_frontier',
            'right_censored_at_additional_step_cap', 'exact_replay',
            'terminal_checkpoint', 'additional_full_body_summary',
            'additional_records')}


def focused_result(result):
    return {key: result[key] for key in ('pre_state_sha256', 'dt_s',
            'old_rejected_reason', 'old_rejected_detail',
            'old_rollback_exact', 'endpoint_accepted',
            'endpoint_exact_replay', 'endpoint_state_sha256',
            'endpoint_final_minJ', 'endpoint_final_global_gap_m',
            'endpoint_final_global_normal_rate_m_s',
            'endpoint_final_closest_u', 'endpoint_final_closest_face',
            'rollback_negative_controls')}


def geometry_result(result):
    return {key: result[key] for key in ('absolute_step_start',
            'absolute_step_end', 'accepted_steps_audited',
            'first_endpoint_global_feature_step',
            'minimum_endpoint_barycentric_weight',
            'minimum_selected_closest_barycentric_weight',
            'minimum_global_closest_barycentric_weight',
            'minimum_global_gap_m', 'maximum_abs_global_normal_rate_m_s',
            'all_selected_face_interior_and_two_endpoint_rows_admitted',
            'all_global_contact_faces_selected', 'rows')}


def identity_result(result):
    return {key: result[key] for key in ('actual_replayed_steps',
            'all_2059_owner_buffers_identical_per_step',
            'all_10240_stress_and_2059_force_hashes_identical_per_step',
            'shared_physical_metric_keys',
            'shared_physical_metric_values_equal_per_step',
            'complete_state_hashes_differ_due_to_witness_ledger_and_cache',
            'rows')}


def run_python(source, args, working_root, log):
    with log.open('wb') as out:
        run = subprocess.run([sys.executable, str(source), *args],
                             cwd=working_root, stdout=out,
                             stderr=subprocess.STDOUT, check=False)
    pathlib.Path(str(log) + '.exit').write_text(str(run.returncode) + '\n')
    if run.returncode != 0:
        raise RuntimeError(f'{source.name} rejected; inspect {log}')


def main():
    began = time.time()
    manifest_path = HERE / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    assert sha(pathlib.Path(__file__)) == manifest['reproducer_sha256']
    roots = [p for p in (HERE, *HERE.parents)
             if (p / 'docs/assets/elastic-yarn-fem-body/payload-manifest.json').is_file()]
    if not roots:
        raise RuntimeError('Run within a repository containing the public full FEM pack.')
    repo = roots[0]
    public = repo / 'docs/assets/elastic-yarn-fem-body'
    fixture = repo / 'docs/assets/elastic-yarn-fem-frame38-provenance.json'
    assert sha(public / 'payload-manifest.json') == manifest['public_payload_manifest_sha256']
    assert sha(fixture) == manifest['public_fixture_sha256']
    for name, expected in manifest['sources_sha256'].items():
        assert sha(HERE / name) == expected, ('source changed', name)
    for name, expected in manifest['owner_candidate_files_sha256'].items():
        assert sha(HERE / 'owner-candidate' / name) == expected
    for name, expected in manifest['owner_continuation_files_sha256'].items():
        assert sha(HERE / 'owner-continuation' / name) == expected
    for name, expected in manifest['old_frontier_files_sha256'].items():
        assert sha(HERE / 'old-frontier' / name) == expected
    for name, expected in manifest['owner_controls_sha256'].items():
        assert sha(HERE / name) == expected

    payload = json.loads((public / 'payload-manifest.json').read_text())
    decoded = {}
    for name, entry in payload['payloads'].items():
        compressed = (public / entry['compressed_file']).read_bytes()
        assert hashlib.sha256(compressed).hexdigest() == entry['compressed_sha256']
        raw = gzip.decompress(compressed)
        assert len(raw) == entry['bytes']
        assert hashlib.sha256(raw).hexdigest() == entry['sha256']
        decoded[name] = raw
    assert len(decoded) == 9
    frozen = {}
    for name, expected in manifest['frozen_physical_source_sha256'].items():
        target = public / 'frozen' / name
        assert sha(target) == expected
        frozen[name] = target.read_bytes()
    assert len(frozen) == 7

    build = repo / 'build'
    build.mkdir(exist_ok=True)
    work = pathlib.Path(tempfile.mkdtemp(prefix='elastic-yarn-endpoint-public-replay-',
                                         dir=build))
    pack = work / 'build/elastic-yarn-fem-step-preparation'
    step = work / 'build/elastic-yarn-fem-endpoint-step'
    old = work / 'build/elastic-yarn-fem-multistep-short/candidate-20us'
    public_fixture = work / 'docs/assets/elastic-yarn-fem-frame38-provenance.json'
    pack.mkdir(parents=True)
    step.mkdir(parents=True)
    old.mkdir(parents=True)
    public_fixture.parent.mkdir(parents=True)
    for name, raw in decoded.items():
        (pack / name).write_bytes(raw)
    for name, raw in frozen.items():
        target = pack / 'frozen' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    shutil.copyfile(fixture, public_fixture)
    for name in manifest['sources_sha256']:
        shutil.copyfile(HERE / name, step / name)
    for name in manifest['old_frontier_files_sha256']:
        shutil.copyfile(HERE / 'old-frontier' / name, old / name)

    # Fresh receipts bind the *actual* supplied public physical inputs. They
    # do not recreate or claim unavailable historical native binary hashes.
    supplied = {str(q.relative_to(work)): sha(q) for q in pack.rglob('*')
                if q.is_file()}
    pack_receipt = {'schema': 'numi.elastic-yarn.public-physical-input-preparation.v1',
                    'sha256': supplied,
                    'historical_native_checkpoint_identity_claimed': False,
                    'authoring_binary_rebuilt_here': False,
                    'inputs_are_reviewed_lossless_public_payloads': True,
                    'source_revision': '69736313414d3afd81f3b181116870143c13966e'}
    (pack / 'receipt.json').write_text(json.dumps(pack_receipt, indent=2) + '\n')
    evidence = {'schema': 'numi.elastic-yarn.public-physical-input-evidence.v1',
                'outputs_sha256': {str(pack / name): hashlib.sha256(raw).hexdigest()
                                   for name, raw in decoded.items()},
                'compressed_payloads_verified': 9,
                'frozen_physical_sources_verified': 7,
                'provenance_receipt_changed_physical_values_unchanged': True}
    (pack / 'evidence.json').write_text(json.dumps(evidence, indent=2) + '\n')
    launch = {'schema': 'numi.elastic-yarn.endpoint-public-fresh-launch.v1',
              'fresh_root': str(work),
              'manifest_sha256': sha(manifest_path),
              'runner_sha256': sha(pathlib.Path(__file__)),
              'public_payload_manifest_sha256': sha(public / 'payload-manifest.json'),
              'new_pack_receipt_sha256': sha(pack / 'receipt.json'),
              'new_pack_evidence_sha256': sha(pack / 'evidence.json'),
              'actual_input_sha256': {str(q.relative_to(work)): sha(q)
                                      for q in work.rglob('*') if q.is_file()},
              'historical_native_checkpoint_identity_claimed': False}
    (work / 'launch.json').write_text(json.dumps(launch, indent=2) + '\n')

    candidate = step / 'candidate-20us'
    run_python(step / 'trajectory.py', ['--dt-us', '20', '--max-steps', '32',
                                       '--interval-us', '320', '--output', str(candidate)],
               work, work / 'trajectory.log')
    reproduced = json.loads((candidate / 'result.json').read_text())
    owner = json.loads((HERE / 'owner-candidate/result.json').read_text())
    assert reproduced['source_sha256'] == manifest['sources_sha256']['endpoint_step.py']
    assert reproduced['normal_block_source_sha256'] == manifest['sources_sha256']['step.py']
    assert reproduced['exact_replay'] and reproduced['primary_accepted_steps'] == 32
    assert digest(physical_result(reproduced)) == manifest['owner_physical_result_digest']
    owner_outputs = {}
    for name in manifest['physical_output_files']:
        left = HERE / 'owner-candidate' / name
        right = candidate / name
        assert left.stat().st_size == right.stat().st_size
        assert sha(left) == sha(right)
        owner_outputs[name] = {'sha256': sha(right), 'bytes': right.stat().st_size,
                               'byte_identical_to_owner': True}
    assert len(owner_outputs) == 9
    assert all(sha(pathlib.Path(path)) == expected
               for path, expected in reproduced['input_sha256'].items())

    continuation_dir = step / 'continuation-exact-20us'
    run_python(step / 'continuation_exact.py',
               ['--max-additional-steps', '32', '--output', str(continuation_dir)],
               work, work / 'continuation.log')
    continuation = json.loads((continuation_dir / 'result.json').read_text())
    assert continuation['bit_identical_candidate_dt']
    assert continuation['dt_s'] == reproduced['dt_s']
    assert continuation['preceding_32step_terminal_state_sha256'] == reproduced['terminal_checkpoint']['complete_state_sha256']
    assert continuation['additional_accepted_steps'] == 32
    assert continuation['total_accepted_steps'] == 64
    assert continuation['exact_replay'] and continuation['right_censored_at_additional_step_cap']
    assert digest(continuation_result(continuation)) == manifest['owner_continuation_physical_result_digest']
    continuation_outputs = {}
    for name in manifest['continuation_physical_output_files']:
        left = HERE / 'owner-continuation' / name
        right = continuation_dir / name
        assert left.stat().st_size == right.stat().st_size
        assert sha(left) == sha(right)
        continuation_outputs[name] = {'sha256': sha(right), 'bytes': right.stat().st_size,
                                      'byte_identical_to_owner': True}
    assert len(continuation_outputs) == 4
    assert all(sha(pathlib.Path(path)) == expected
               for path, expected in continuation['input_sha256'].items())

    # Reproduce the old fixed-centroid false rejection from its complete
    # retained 12-step state, then verify the same exact-dt endpoint step.
    run_python(step / 'focused_control.py', [], work, work / 'focused.log')
    focused = json.loads((step / 'focused-control.json').read_text())
    assert focused['passed'] and digest(focused_result(focused)) == manifest['owner_focused_digest']
    assert focused['old_rejected_reason'] == 'contact-manifold/gap'
    assert focused['endpoint_accepted'] and focused['endpoint_exact_replay']
    assert all(q['rollback_exact'] and q['complete_state_unchanged']
               for q in focused['rollback_negative_controls'])

    run_python(step / 'geometry_audit.py', ['--input', str(candidate),
                                           '--output', str(step / 'geometry-audit.json')],
               work, work / 'geometry.log')
    geometry = json.loads((step / 'geometry-audit.json').read_text())
    assert geometry['passed'] and digest(geometry_result(geometry)) == manifest['owner_geometry_digest']
    assert geometry['accepted_steps_audited'] == 32
    assert geometry['all_selected_face_interior_and_two_endpoint_rows_admitted']

    run_python(step / 'geometry_audit.py', ['--input', str(continuation_dir),
                                           '--output', str(step / 'continuation-geometry-audit.json')],
               work, work / 'continuation-geometry.log')
    continuation_geometry = json.loads((step / 'continuation-geometry-audit.json').read_text())
    assert continuation_geometry['passed']
    assert digest(geometry_result(continuation_geometry)) == manifest['owner_continuation_geometry_digest']
    assert continuation_geometry['absolute_step_start'] == 33
    assert continuation_geometry['absolute_step_end'] == 64

    # This deeper owner-buffer predecessor identity control costs another
    # ~100 s, but closes the exact first-12 physical-state comparison.
    run_python(step / 'predecessor_identity.py', [], work, work / 'identity.log')
    identity = json.loads((step / 'predecessor-identity.json').read_text())
    assert identity['passed'] and digest(identity_result(identity)) == manifest['owner_identity_digest']
    assert identity['all_2059_owner_buffers_identical_per_step']

    result = {'schema': 'numi.elastic-yarn.endpoint-multistep.portable-public-reproduction.v1',
              'actual_reproduction_exit': 0, 'fresh_root': str(work),
              'public_payloads_verified': 9, 'frozen_physical_sources_verified': 7,
              'new_physical_input_receipt': True,
              'historical_native_checkpoint_identity_claimed': False,
              'owner_physical_result_digest': manifest['owner_physical_result_digest'],
              'fresh_physical_result_digest': digest(physical_result(reproduced)),
              'nine_initial_output_files_byte_identical': True,
              'owner_outputs': owner_outputs,
              'four_continuation_output_files_byte_identical': True,
              'continuation_outputs': continuation_outputs,
              'linked_complete_states_accepted': 64,
              'linked_elapsed_s': continuation['terminal_checkpoint']['elapsed_s'],
              'focused_old_negative_and_endpoint_positive_reproduced': True,
              'all_64_contact_features_audited': True,
              'all_12_predecessor_owner_buffers_identical': True,
              'matched_interval_s': reproduced['matched_interval']['interval_s'],
              'full_cumulative_energy_residual_J': reproduced['matched_interval']['full_cumulative_energy_residual_J'],
              'half_cumulative_energy_residual_J': reproduced['matched_interval']['half_cumulative_energy_residual_J'],
              'linked_64step_cumulative_energy_residual_J':
                  reproduced['full_body_summary']['cumulative_physical_energy_residual_J'] +
                  continuation['additional_full_body_summary']['cumulative_physical_energy_residual_J'],
              'trajectory_exit_marker_sha256': sha(work / 'trajectory.log.exit'),
              'continuation_exit_marker_sha256': sha(work / 'continuation.log.exit'),
              'focused_exit_marker_sha256': sha(work / 'focused.log.exit'),
              'geometry_exit_marker_sha256': sha(work / 'geometry.log.exit'),
              'continuation_geometry_exit_marker_sha256': sha(work / 'continuation-geometry.log.exit'),
              'identity_exit_marker_sha256': sha(work / 'identity.log.exit'),
              'actual_runtime_s': time.time() - began,
              'scope': 'New FP64 split of complete represented body and local two-node yarn; numerical contact, no CCD/native/full-scene/energy-closure or temporal-convergence claim.'}
    (work / 'reproduction-result.json').write_text(json.dumps(result, indent=2) + '\n')
    (HERE / 'reproduction-result.json').write_text(json.dumps(result, indent=2) + '\n')
    (HERE / 'fresh-launch.json').write_text(json.dumps(launch, indent=2) + '\n')
    print(json.dumps({'actual_exit': 0, 'fresh_root': str(work),
                      'output_files_identical': len(owner_outputs) + len(continuation_outputs),
                      'old_negative': focused['old_rejected_reason'],
                      'first12_owner_identity': True,
                      'actual_runtime_s': result['actual_runtime_s']}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
