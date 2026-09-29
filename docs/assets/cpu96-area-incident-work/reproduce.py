#!/usr/bin/env python3
"""Portable CPU-only reproduction of the CPU96 material-face work blocker."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile

HERE=Path(__file__).resolve().parent
SEALED_TOPOLOGY='1641702cf95a7b4871b422fb5e859bd1118e324f9c3c1cbd9f33b70399c9944f'
SEALED_COMMON='7927f6b3f0267dacd8d78ee883c776865689eb008108d1a212427f9eec4547b5'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def command(run,name,argv,expected):
    p=subprocess.run(argv,cwd=run,capture_output=True,text=True)
    (run/(name+'.log')).write_text(p.stdout+p.stderr)
    (run/(name+'.exit')).write_text(str(p.returncode)+'\n')
    if p.returncode!=expected:raise RuntimeError(f'{name} exit {p.returncode}, expected {expected}; see {run/(name+".log")}')
    return p.stdout

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,help='new output directory; default is a system temporary directory')
    args=parser.parse_args()
    manifest=json.loads((HERE/'source-manifest.json').read_text())
    assert manifest['schema']=='numi.cpu96.area-incident-work.portable-inputs.v1'
    for rel,digest in manifest['sha256'].items():
        path=HERE/rel
        if not path.is_file() or sha(path)!=digest:raise RuntimeError(f'input binding failed: {rel}')
    if args.output_dir:
        parent=args.output_dir.resolve()
        parent.mkdir(parents=True,exist_ok=False)
    else:parent=Path(tempfile.mkdtemp(prefix='numi-cpu96-area-work-'))
    run=parent/'asset'
    shutil.copytree(HERE,run,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    before={rel:sha(run/rel) for rel in manifest['sha256']}
    command(run,'compile',['clang++','-std=c++20','-O1','-I','frozen/include','topology.cpp','-o','topology'],0)
    topology_stdout=command(run,'topology',[str(run/'topology')],0)
    (run/'topology.json').write_text(topology_stdout)
    assert sha(run/'topology.json')==SEALED_TOPOLOGY
    common_stdout=command(run,'common-event',[sys.executable,'common_event.py'],0)
    assert sha(run/'common-event.json')==SEALED_COMMON
    negative_stdout=command(run,'negative-fullscene',[sys.executable,'common_event.py','--claim-repaired-drop'],1)
    assert sha(run/'common-event.json')==SEALED_COMMON
    negative=json.loads(negative_stdout.splitlines()[-1])
    assert negative['classification']=='REJECTED' and negative['local_passivity_cannot_qualify_CPU96']
    topo=json.loads((run/'topology.json').read_text())
    event=json.loads((run/'common-event.json').read_text())
    assert topo['failed_face_owners']==[1433,1444,1445] and len(event['incident_faces'])==15
    assert topo['incident_distance_constraints']==10 and topo['incident_bend_constraints']==14 and topo['incident_knot_constraints']==10
    assert event['selected_only_response_accepted'] and event['post_selected_response_all_incident_faces_clear']
    assert event['selected_response_local_kinetic_change_J']<0<event['diagnostic_incident_distance_potential_change_J']
    after={rel:sha(run/rel) for rel in manifest['sha256']}
    assert before==after
    evidence={
      'schema':'numi.cpu96.area-incident-work.public-reproduction.v1',
      'source_manifest_sha256':sha(HERE/'source-manifest.json'),
      'actual_exits':{'compile':0,'topology':0,'common-event':0,'negative-fullscene':1},
      'sealed_topology_sha256':SEALED_TOPOLOGY,'sealed_common_event_sha256':SEALED_COMMON,
      'inputs_identical_before_after':True,
      'topology':{k:topo[k] for k in ('cloth_nodes','render_faces','failed_face_owners','failed_face_masses_kg','incident_distance_constraints','incident_bend_constraints','incident_knot_constraints','incident_local_node_pairs','incident_yarn_segments','distance_graph_reachable_from_failed_owner')},
      'synthetic_chord':{'incident_faces':len(event['incident_faces']),'earliest_face':event['earliest_incident_face']['face'],
        'local_kinetic_change_J':event['selected_response_local_kinetic_change_J'],
        'incident_distance_spring_proxy_change_J':event['diagnostic_incident_distance_potential_change_J'],
        'all_incident_post_paths_clear':event['post_selected_response_all_incident_faces_clear']},
      'scope':'synthetic CPU geometry/work proxy; no actual CPU96 repair, complete physical energy closure, production integration or GPU run'
    }
    (run/'reproduction.json').write_text(json.dumps(evidence,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'output_dir':str(run),'evidence':evidence},sort_keys=True))
    return 0

if __name__=='__main__':raise SystemExit(main())
