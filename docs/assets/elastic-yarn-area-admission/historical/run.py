#!/usr/bin/env python3
"""Retain actual bounded CPU exits and source-bound evidence; no native launch."""
import datetime,hashlib,json,platform,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;STUDY=HERE.parent;ROOT=STUDY.parent.parent
def binding(path):return {'path':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
def main():
 source_names=('area_admission.py','qualify.py','README.md','run.py')
 frozen_sources={name:binding(HERE/name) for name in source_names}
 actual={};runs=('qualification','moving-pin','high-common-y','high-common-x','position-reconstruction','local-rate-only','tangent-margin')
 for name in runs:
  command=[sys.executable,str(HERE/'qualify.py')]
  if name!='qualification':command+=['--negative',name]
  p=subprocess.run(command,capture_output=True,text=True,cwd=ROOT)
  log=p.stdout+p.stderr;(HERE/(name+'.log')).write_text(log);(HERE/(name+'.exit')).write_text(str(p.returncode)+'\n');actual[name]=p.returncode
  expected=0 if name=='qualification' else 1
  if p.returncode!=expected:raise RuntimeError(f'{name} actual exit {p.returncode}, expected {expected}; retained log')
 for name,b in frozen_sources.items():assert binding(HERE/name)==b
 result=json.loads((HERE/'result.json').read_text());metrics={k:v for k,v in result.items() if k not in ('temporal_cases','domain_controls','passive_controls','rejected_mechanisms')}
 paths=list(source_names)+['result.json']+[name+suffix for name in runs for suffix in ('.log','.exit')]
 receipt={'schema':'numi.loaded-drop.preventive-material-face-area-admission.v1','timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
  'device':'Apple CPU only, Python FP64 owners / exact Fraction / independent Decimal100 analytic roots and exact Bernstein bounds',
  'python':sys.version,'platform':platform.platform(),'actual_exits':actual,'bindings':{name:binding(HERE/name) for name in paths},
  'input_bindings':binding(STUDY/'input-bindings.json'),'sealed_first_study':{name:binding(STUDY/name) for name in ('study.py','evidence.json','qualification.log','result.json','README.md')},
  'metrics':metrics,'limits':['no production source, frozen input, GPU, full-scene or full-trajectory changes',
   'frame620 rest-to-failure chord is declared synthetic; owner velocities/history are absent',
   'exact continuous affine paths from represented FP64 endpoints; no native integrator rounding proof',
   '128 shared temporal proof nodes plus bounded degree4 rational work and bit complexity; 16 boundary retries',
   'fixed zero-velocity pins only; moving support work rejected',
   'strict unchanged1e-8 area and30m/s source domain; no material law or stronger area target',
   'later post-response crossing and represented double-root margin remain explicit unresolved',
   'local kinetic passivity and reciprocal ledgers do not close all yarn/constraint/external energy',
   'full corrected CPU96 drop remains FAIL; no actual repair qualification']}
 (HERE/'evidence.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
 # Final source copies provide a stable review target without changing original inputs.
 frozen=HERE/'frozen';frozen.mkdir(exist_ok=True)
 for name in source_names+('evidence.json','result.json'):
  target=frozen/name;data=(HERE/name).read_bytes()
  if target.exists() and target.read_bytes()!=data:raise RuntimeError('sealed preventive artifact differs; use a new revision path')
  target.write_bytes(data)
 print(json.dumps({'actual_exits':actual,'evidence_sha256':binding(HERE/'evidence.json')['sha256'],'core_sha256':binding(HERE/'area_admission.py')['sha256'],'qualifier_sha256':binding(HERE/'qualify.py')['sha256'],'maximum_temporal_nodes':metrics['maximum_temporal_proof_nodes']},sort_keys=True))
if __name__=='__main__':main()
