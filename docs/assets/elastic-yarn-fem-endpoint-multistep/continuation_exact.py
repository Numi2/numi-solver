#!/usr/bin/env python3
"""Exact-binary-dt replay continuation from the complete 32-step checkpoint."""
import argparse
import copy
import gzip
import importlib.util
import json
import pathlib
import struct
import sys
import time

P = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('endpoint_trajectory', P / 'trajectory.py')
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)
E = T.N
B = T.B


def load_complete_checkpoint(directory, binding):
    state = json.loads(gzip.decompress((directory / binding['metadata_file']).read_bytes()))
    records = list(struct.iter_unpack('<6d', (directory / binding['position_file']).read_bytes()))
    assert len(records) == 2059
    state['x'] = [list(q[:3]) for q in records]
    state['v'] = [list(q[3:]) for q in records]
    assert B.state_sha(state) == binding['complete_state_sha256']
    assert B.sha(directory / binding['position_file']) == binding['position_sha256']
    assert B.sha(directory / binding['metadata_file']) == binding['metadata_sha256']
    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-additional-steps', type=int, default=32)
    parser.add_argument('--output', type=pathlib.Path, default=P / 'continuation-exact-20us')
    args = parser.parse_args()
    assert 1 <= args.max_additional_steps <= 128
    began = time.time()
    source = P / 'candidate-20us'
    preceding = json.loads((source / 'result.json').read_text())
    assert preceding['exact_replay'] and preceding['primary_accepted_steps'] == 32
    assert preceding['right_censored_at_max_steps']
    assert preceding['source_sha256'] == T.SOURCE_SHA
    dt = preceding['dt_s']
    assert dt == 20.0 * 1e-6
    start = load_complete_checkpoint(source, preceding['terminal_checkpoint'])
    assert start['steps'] == 32 and abs(start['elapsed'] - 32 * dt) < 1e-18
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    class CheckpointModel(E.Model):
        def initial(self):
            return copy.deepcopy(start)

    primary = T.run_pass('primary', CheckpointModel(), dt,
                         args.max_additional_steps, output)
    accepted = len(primary['records'])
    terminal = primary['state']
    binding = T.save_full_state(output / 'terminal', terminal)
    repeat = T.run_pass('replay', CheckpointModel(), dt,
                        accepted + (primary['frontier'] is not None), output,
                        expected=primary['records'])
    exact = (repeat['records'] == primary['records'] and
             repeat['frontier'] == primary['frontier'] and
             B.state_sha(repeat['state']) == B.state_sha(terminal))
    assert exact
    assert terminal['steps'] == 32 + accepted
    frontier = primary['frontier']
    if frontier is not None:
        frontier['absolute_attempted_step'] = 32 + frontier['attempted_step']
        assert frontier['state_sha256'] == B.state_sha(terminal)
        assert frontier['rollback_exact']
    result = {'schema': 'numi.elastic-yarn.endpoint-normal-block-checkpoint-continuation.v1',
              'preceding_result_sha256': B.sha(source / 'result.json'),
              'preceding_receipt_sha256': B.sha(source / 'receipt.json'),
              'preceding_32step_terminal_state_sha256': preceding['terminal_checkpoint']['complete_state_sha256'],
              'input_sha256': E.Model().inputs,
              'source_sha256': {str(q): B.sha(q) for q in
                                (P / 'endpoint_step.py', P / 'step.py',
                                 P / 'centroid_step.py', P / 'trajectory.py', pathlib.Path(__file__))},
              'dt_s': dt, 'bit_identical_candidate_dt': True,
              'additional_accepted_steps': accepted,
              'total_accepted_steps': terminal['steps'],
              'first_unsupported_frontier': frontier,
              'right_censored_at_additional_step_cap': frontier is None,
              'exact_replay': exact,
              'terminal_checkpoint': binding,
              'additional_full_body_summary': T.summarize(primary['records']),
              'additional_records': primary['records'],
              'primary_log_sha256': primary['log_sha256'],
              'replay_log_sha256': repeat['log_sha256'],
              'runtime_s': time.time() - began,
              'limits': ['Continuation from complete original 32-step CPU state with unchanged numerical geometry/J/rest/contact gates.',
                         'All additional 2059 owner, 10240 tet stress/J/U/force and 1280 face/normal checks are executed per accepted step.',
                         'First unsupported event rejects and leaves complete accepted state unchanged.',
                         'FP64 numerical boundary oracle, no CCD, no native trajectory or temporal convergence claim.',
                         'Physical energy residual remains open and measured.']}
    target = output / 'result.json'
    target.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    receipt = {'schema': 'numi.elastic-yarn.endpoint-continuation-source-bound-receipt.v1',
               'actual_program_exit': 0, 'result_sha256': B.sha(target),
               'source_sha256': result['source_sha256'],
               'input_sha256': result['input_sha256'],
               'retained_outputs': {q.name: {'sha256': B.sha(q), 'bytes': q.stat().st_size}
                                    for q in output.iterdir() if q.is_file() and q.name != 'receipt.json'}}
    (output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'additional_accepted': accepted,
                      'total_accepted': terminal['steps'],
                      'frontier': frontier['reason'] if frontier else None,
                      'absolute_frontier_step': frontier['absolute_attempted_step'] if frontier else None,
                      'exact_replay': exact, 'runtime_s': result['runtime_s']}), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
