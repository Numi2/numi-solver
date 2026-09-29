#!/usr/bin/env python3
"""Reproduce the bounded CPU area-admission study from this package alone.

No native build, GPU, historical binary, or external dataset is needed.
Outputs are written to a fresh private directory, never into these assets.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile

HERE=Path(__file__).resolve().parent
NEGATIVES=('moving-pin','high-common-y','high-common-x','position-reconstruction',
           'local-rate-only','tangent-margin')

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def run_command(directory,name,command,expected):
    p=subprocess.run(command,cwd=directory,capture_output=True,text=True)
    (directory/(name+'.log')).write_text(p.stdout+p.stderr)
    (directory/(name+'.exit')).write_text(str(p.returncode)+'\n')
    if p.returncode!=expected:
        raise RuntimeError(f'{name}: actual exit {p.returncode}, expected {expected}; see {directory/name}.log')
    return p.stdout

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,help='fresh destination directory; default is a system temporary directory')
    args=parser.parse_args()
    manifest=json.loads((HERE/'source-manifest.json').read_text())
    assert manifest['schema']=='numi.elastic-yarn-area-admission.portable-inputs.v1'
    for rel,digest in manifest['sha256'].items():
        path=HERE/rel
        if not path.is_file() or sha(path)!=digest:raise RuntimeError(f'package source binding failed: {rel}')
    if args.output_dir:
        out=args.output_dir.resolve()
        out.mkdir(parents=True,exist_ok=False)
    else:out=Path(tempfile.mkdtemp(prefix='numi-area-admission-'))
    run=out/'asset';shutil.copytree(HERE,run,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    source_before={rel:sha(run/rel) for rel in manifest['sha256']}
    qualification=run_command(run,'qualification',[sys.executable,'qualify.py'],0)
    historical=json.loads((run/'historical/result.json').read_text())
    reproduced=json.loads((run/'result.json').read_text())
    removed=('frozen_binding_checks_before','frozen_binding_checks_after')
    assert all(reproduced[k]==historical[k] for k in historical if k not in removed)
    assert reproduced['frozen_binding_checks_before']==4
    assert reproduced['frozen_binding_checks_after']==4
    assert historical['frozen_binding_checks_before']==17
    assert historical['frozen_binding_checks_after']==17
    negative_exits={}
    for name in NEGATIVES:
        stdout=run_command(run,name,[sys.executable,'qualify.py','--negative',name],1)
        row=json.loads(stdout)
        assert row['qualification']=='REJECTED_OR_UNRESOLVED' and row['negative_control']==name
        negative_exits[name]=1
    frozen_stdout=run_command(run,'frozen-equality',[sys.executable,'strict_gate_check.py','frozen'],1)
    candidate_stdout=run_command(run,'candidate-equality',[sys.executable,'strict_gate_check.py','candidate'],0)
    frozen_row=json.loads(frozen_stdout.splitlines()[0]);candidate_row=json.loads(candidate_stdout.splitlines()[0])
    assert frozen_row['accepted'] and not candidate_row['accepted']
    independent_stdout=run_command(run,'independent-review',[sys.executable,'independent_review.py'],0)
    independent=json.loads((run/'independent-result.json').read_text())
    archived=json.loads((run/'historical/independent-result.json').read_text())
    assert all(independent[k]==archived[k] for k in archived if k!='candidate_qualification_source_sha256')
    assert independent['status']=='PASS_WITH_ONE_FROZEN_HELPER_DEFECT_CORRECTED_LOCALLY'
    source_after={rel:sha(run/rel) for rel in manifest['sha256']}
    assert source_after==source_before
    metrics={key:reproduced[key] for key in ('named_temporal_cases','analytic_planar_first_root_cases',
        'anchored_response_cases','safe_linear_contraction_cases','general_3d_independent_bernstein_clear_cases',
        'ordinary_represented_velocity_cases','ordinary_velocity_admitted','maximum_temporal_proof_nodes')}
    evidence={
      'schema':'numi.elastic-yarn-area-admission.portable-reproduction.v1',
      'source_manifest_sha256':sha(HERE/'source-manifest.json'),
      'actual_exits':{'qualification':0,'frozen-equality':1,'candidate-equality':0,'independent-review':0,**negative_exits},
      'historical_case_result_identical_except_external_binding_counts':True,
      'historical_external_binding_checks':17,'portable_included_binding_checks':4,
      'metrics':metrics,'independent_review_sha256':sha(run/'independent-result.json'),
      'qualification_result_sha256':sha(run/'result.json'),
      'inputs_identical_before_after':True,
      'scope':'CPU exact affine triangle primitive only; synthetic selected-owner chord, no actual CPU96 repair, GPU, production integration, native history or incident-work closure'
    }
    (run/'reproduction.json').write_text(json.dumps(evidence,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'output_dir':str(run),'evidence':evidence},sort_keys=True))
    return 0

if __name__=='__main__':raise SystemExit(main())
