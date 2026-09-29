#!/usr/bin/env python3
"""Flip exactly one bundled source byte and prove manifest rejection is prephysics."""
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents
            if (p / 'docs/assets/elastic-yarn-fem-body/payload-manifest.json').is_file())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest = json.loads((HERE / 'manifest.json').read_text())
    assert (HERE / 'reproduction.exit').read_text().strip() == '0'
    before = {p.name for p in (REPO / 'build').glob('elastic-yarn-endpoint-public-replay-*')}
    negative = pathlib.Path(tempfile.mkdtemp(prefix='elastic-yarn-endpoint-prephysics-negative-',
                                             dir=REPO / 'build'))
    shutil.copytree(HERE, negative, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('__pycache__'))
    source = negative / 'endpoint_step.py'
    raw = source.read_bytes()
    offset = raw.index(b'The old block')
    assert raw[offset:offset + 1] == b'T'
    mutated = raw[:offset] + b'X' + raw[offset + 1:]
    assert len(mutated) == len(raw)
    assert sum(a != b for a, b in zip(raw, mutated)) == 1
    source.write_bytes(mutated)
    assert hashlib.sha256(raw).hexdigest() == manifest['sources_sha256']['endpoint_step.py']
    assert sha(source) != manifest['sources_sha256']['endpoint_step.py']
    log = HERE / 'negative-manifest-mutation.log'
    with log.open('wb') as output:
        run = subprocess.run([sys.executable, str(negative / 'reproduce.py')],
                             cwd=REPO, stdout=output,
                             stderr=subprocess.STDOUT, check=False)
    marker = HERE / 'negative-manifest-mutation.exit'
    marker.write_text(str(run.returncode) + '\n')
    after = {p.name for p in (REPO / 'build').glob('elastic-yarn-endpoint-public-replay-*')}
    message = log.read_text()
    assert run.returncode != 0 and "('source changed', 'endpoint_step.py')" in message
    assert before == after, 'physics fresh root unexpectedly launched'
    result = {'schema': 'numi.elastic-yarn.endpoint-public-manifest-negative.v1',
              'control_passed': True,
              'negative_copy_root': str(negative),
              'negative_controller_source_sha256': sha(pathlib.Path(__file__)),
              'source_name': 'endpoint_step.py',
              'original_source_sha256': hashlib.sha256(raw).hexdigest(),
              'mutated_source_sha256': sha(source),
              'changed_byte_offset': offset,
              'changed_byte_from_hex': '54',
              'changed_byte_to_hex': '58',
              'changed_byte_count': 1,
              'mutated_runner_actual_exit': run.returncode,
              'rejected_before_physics_root_creation': True,
              'physics_roots_before': sorted(before),
              'physics_roots_after': sorted(after),
              'failure_log_sha256': sha(log),
              'failure_exit_marker_sha256': sha(marker),
              'failure_reason': 'Manifest source hash mismatch for endpoint_step.py'}
    (HERE / 'negative-manifest-mutation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'control_passed': True,
                      'mutated_runner_actual_exit': run.returncode,
                      'changed_byte_count': 1,
                      'rejected_before_physics_root_creation': True}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
