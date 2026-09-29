#!/usr/bin/env python3
"""Read-only common event audit on a declared synthetic frame-0->620 chord."""
from fractions import Fraction as Q
from decimal import Decimal, localcontext
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE
SOURCE=HERE/'area_admission.py'
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()=='bc83f8c623e57796cb1d3ecc92830d270fc3d121ae56206675188ba48d825fad'
spec=importlib.util.spec_from_file_location('area_admission_common_event',SOURCE)
a=importlib.util.module_from_spec(spec);sys.modules[spec.name]=a;spec.loader.exec_module(a)

def obj(name,expected):
    path=HERE/'frozen'/name
    assert hashlib.sha256(path.read_bytes()).hexdigest()==expected
    positions=[];faces=[]
    for line in path.read_text().splitlines():
        if line.startswith('v '):positions.append(tuple(map(float,line.split()[1:4])))
        elif line.startswith('f '):faces.append(tuple(int(z.split('/')[0])-1 for z in line.split()[1:4]))
    return positions,faces

def main():
    start,faces0=obj('frame0.obj','21483357b0deb815b8cb071e40c171b46f05bd285425d0ed4661e2c0d68b3122')
    end,faces1=obj('frame620.obj','e52a1c9cc7d56d0661cc013e2eaf9345da55539de3537536dd98be86a4b71fa3')
    assert faces0==faces1 and len(faces0)==2880 and faces0[2813]==(1433,1444,1445)
    selected=set(faces0[2813]);records=[]
    for index,face in enumerate(faces0):
        if not selected.intersection(face):continue
        m=a.Motion(tuple(start[i] for i in face),tuple(end[i] for i in face))
        result=a.cast(m)
        record={'face':index,'owners':face,'status':result['status'],'accepted':result['accepted'],
                'rest_area_m2':a.represented_area(m.start),'end_area_m2':a.represented_area(m.end),
                'nodes':result['total_proof_nodes']}
        if result['status']=='first_boundary':
            record.update(lower_time_exact=result['lower_time_exact'],upper_time_exact=result['upper_time_exact'],root_count=result['root_count'])
        records.append(record)
    roots=[r for r in records if r['status']=='first_boundary']
    assert len(records)==15 and roots
    roots.sort(key=lambda r:Q(r['upper_time_exact']))
    selected_row=next(r for r in records if r['face']==2813)
    topology=json.loads((HERE/'topology.json').read_text())
    assert topology['failed_face_owners']==list(faces0[2813]) and topology['failed_face_masses_kg']==[5e-5]*3
    local=a.preventive(a.Motion(tuple(start[i] for i in faces0[2813]),tuple(end[i] for i in faces0[2813])),tuple(topology['failed_face_masses_kg']))
    assert local['accepted'] and local['status']=='admitted_response'
    t=Q.from_float(local['boundary_time']);remaining=1-t
    incident_nodes=sorted(set(i for row in records for i in row['owners']))
    boundary={i:tuple(float(Q.from_float(x)+t*(Q.from_float(y)-Q.from_float(x)))
                      for x,y in zip(start[i],end[i])) for i in incident_nodes}
    assert tuple(boundary[i] for i in faces0[2813])==tuple(tuple(p) for p in local['boundary_positions'])
    velocities={i:tuple(float(Q.from_float(y)-Q.from_float(x)) for x,y in zip(start[i],end[i]))
                for i in incident_nodes}
    for i,v in zip(faces0[2813],local['response']['velocities']):velocities[i]=tuple(v)
    changed={i:tuple(float(Q.from_float(x)+Q.from_float(v)*remaining)
                     for x,v in zip(boundary[i],velocities[i])) for i in incident_nodes}
    post=[]
    for row in records:
        f=row['owners'];r=a.cast(a.Motion(tuple(boundary[i] for i in f),tuple(changed[i] for i in f)))
        post.append({'face':row['face'],'status':r['status'],'accepted':r['accepted'],
                     'end_area_m2':a.represented_area(tuple(changed[i] for i in f))})
    def potential(rows,positions):
        total=0.
        for row in rows:
            if len(row)==5:_,i,j,rest,compliance=row
            else:_,i,_,j,rest,_,compliance=row
            if i not in positions or j not in positions:continue
            distance=math.dist(positions[i],positions[j]);total+=(distance-rest)**2/(2*compliance)
        return total
    def decimal_potential(rows,positions):
        with localcontext() as ctx:
            ctx.prec=80
            total=Decimal(0)
            for row in rows:
                if len(row)==5:_,i,j,rest,compliance=row
                else:_,i,_,j,rest,_,compliance=row
                if i not in positions or j not in positions:continue
                delta=[Decimal.from_float(positions[i][k])-Decimal.from_float(positions[j][k])
                       for k in range(3)]
                distance=sum((z*z for z in delta),Decimal(0)).sqrt()
                strain=distance-Decimal.from_float(rest)
                total+=strain*strain/(2*Decimal.from_float(compliance))
            return total
    original_endpoint={i:end[i] for i in incident_nodes}
    distance_original=potential(topology['incident_distance_rows'],original_endpoint)
    distance_modified=potential(topology['incident_distance_rows'],changed)
    bend_original=potential(topology['incident_bend_rows'],original_endpoint)
    bend_modified=potential(topology['incident_bend_rows'],changed)
    decimal_distance_delta=(decimal_potential(topology['incident_distance_rows'],changed)-
                            decimal_potential(topology['incident_distance_rows'],original_endpoint))
    decimal_bend_delta=(decimal_potential(topology['incident_bend_rows'],changed)-
                        decimal_potential(topology['incident_bend_rows'],original_endpoint))
    assert decimal_distance_delta>0 and abs(float(decimal_distance_delta)-(distance_modified-distance_original))<1e-10
    assert abs(float(decimal_bend_delta)-(bend_modified-bend_original))<1e-10
    out={'scope':'synthetic full-frame linear chord only; not actual CPU96 substep/history',
         'incident_faces':records,'earliest_incident_face':roots[0],
         'failed_face_2813':selected_row,'selected_only_response_status':local['status'],
         'selected_only_response_accepted':local['accepted'],
         'post_selected_response_incident_faces':post,
         'post_selected_response_all_incident_faces_clear':all(row['accepted'] and row['status']=='clear' for row in post),
         'selected_response_local_kinetic_change_J':local['response']['represented_kinetic_change_J'],
         'diagnostic_incident_distance_potential_change_J':distance_modified-distance_original,
         'diagnostic_incident_bend_potential_change_J':bend_modified-bend_original,
         'independent_decimal80_incident_distance_potential_change_J':str(decimal_distance_delta),
         'independent_decimal80_incident_bend_potential_change_J':str(decimal_bend_delta),
         'diagnostic_potential_scope':'spring proxy from authored rest lengths/chords and compliances only; no total XPBD/contact/grip work authority',
         'earliest_is_selected_face':roots[0]['face']==2813,
         'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest()}
    (HERE/'common-event.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'incident_faces':len(records),'root_faces':len(roots),
        'earliest_face':roots[0]['face'],'earliest_upper':roots[0]['upper_time_exact'],
        'selected_upper':selected_row.get('upper_time_exact'),
        'selected_only_response':local['status'],
        'post_all_incident_clear':out['post_selected_response_all_incident_faces_clear'],
        'local_kinetic_change_J':out['selected_response_local_kinetic_change_J'],
        'incident_distance_potential_change_J':out['diagnostic_incident_distance_potential_change_J'],
        'incident_bend_potential_change_J':out['diagnostic_incident_bend_potential_change_J']}))
    if '--claim-repaired-drop' in sys.argv:
        print(json.dumps({'classification':'REJECTED',
            'reason':'no actual owner velocities/history or full incident constraint/contact work and rollback ledger',
            'local_passivity_cannot_qualify_CPU96':True,
            'diagnostic_distance_potential_change_J':out['diagnostic_incident_distance_potential_change_J']}))
        return 1
    return 0
if __name__=='__main__':raise SystemExit(main())
