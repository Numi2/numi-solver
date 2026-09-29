#!/usr/bin/env python3
"""Bounded CPU qualification; analytic roots do not use the candidate Sturm code."""
import argparse,hashlib,importlib.util,json,math,random,sys
from decimal import Decimal,localcontext
from fractions import Fraction as Q
from pathlib import Path
import area_admission as a
HERE=Path(__file__).resolve().parent;STUDY=HERE/'legacy'
SEALED_SOURCE='9520a130e51d25e270f53efe3366367b283aedc9fceb9b4ed1ea643d92c5770c'
SEALED_RECEIPT='c6adf30bf17d9a6b2c301e03791630638dc47b3d6a6676d07f503b7e5e39da8e'
QG=Q.from_float(a.GATE)
def fq(x):return Q.from_float(float(x))
def d(q):return Decimal(q.numerator)/Decimal(q.denominator)
def norm(x):return math.hypot(*x)
def xprod(x,y):return (x[1]*y[2]-x[2]*y[1],x[2]*y[0]-x[0]*y[2],x[0]*y[1]-x[1]*y[0])
def difference(x,y):return tuple(z-w for z,w in zip(x,y))
def exact_kinetic(v,m):return sum((fq(z)*sum((fq(w)**2 for w in p),Q(0))/2 for p,z in zip(v,m)),Q(0))
def exact_area_squared(x):
 p=tuple(tuple(fq(z) for z in owner) for owner in x);c=xprod(difference(p[1],p[0]),difference(p[2],p[0]))
 return sum((z*z for z in c),Q(0))/4
def independent_polynomial(m):
 # Reconstruct the vector quadratic from determinant samples at 0,1/2,1,
 # independently of the candidate's endpoint-edge coefficient expansion.
 samples=[]
 for t in (Q(0),Q(1,2),Q(1)):
  owners=tuple(tuple(fq(z)+t*(fq(w)-fq(z)) for z,w in zip(x,y)) for x,y in zip(m.start,m.end))
  samples.append(xprod(difference(owners[1],owners[0]),difference(owners[2],owners[0])))
 coefficients=[]
 for k in range(3):
  c0,cm,cend=(sample[k] for sample in samples)
  coefficients.append((c0,4*cm-3*c0-cend,2*cend+2*c0-4*cm))
 p=[Q(0)]*5
 for c in coefficients:
  for i in range(3):
   for j in range(3):p[i+j]+=c[i]*c[j]
 p[0]-=4*QG**2
 return tuple(p)
def independent_bernstein_positive(p):
 # Polynomial convex-hull bound, with a separate32-split CPU oracle budget.
 b=tuple(sum((p[j]*Q(math.comb(k,j),math.comb(4,j)) for j in range(k+1)),Q(0)) for k in range(5))
 stack=[b];used=0
 while stack:
  b=stack.pop()
  if min(b)>0:continue
  used+=1
  if used>32:return False
  levels=[b]
  while len(levels[-1])>1:levels.append(tuple((x+y)/2 for x,y in zip(levels[-1],levels[-1][1:])))
  stack.append(tuple(row[0] for row in levels));stack.append(tuple(row[-1] for row in reversed(levels)))
 return True
def verify_inputs():
 # Portable bindings: the original native binary and complete OBJ histories
 # remain historical SHA commitments, not artifacts revalidated by this pack.
 assert hashlib.sha256((STUDY/'study.py').read_bytes()).hexdigest()==SEALED_SOURCE
 assert hashlib.sha256((STUDY/'evidence.json').read_bytes()).hexdigest()==SEALED_RECEIPT
 assert hashlib.sha256((STUDY/'result.json').read_bytes()).hexdigest()=='fa4dee71bd60cc2c50a806f39d650f5e2e06d3244402057423b1486bf25375a2'
 owners=json.loads((STUDY/'chord-owners.json').read_text())
 assert len(owners['frames'])==2 and all(len(f['owners_1433_1444_1445'])==3 for f in owners['frames'])
 return 4
def old_study():
 spec=importlib.util.spec_from_file_location('sealed_area_study',STUDY/'study.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
def fixtures():
 # Published selected FP64 owners; this is a synthetic chord, not history.
 rows=json.loads((STUDY/'chord-owners.json').read_text())['frames']
 assert [row['source_name'] for row in rows]==['frame0.obj','frame620.obj']
 return tuple(tuple(p) for p in rows[0]['owners_1433_1444_1445']),tuple(tuple(p) for p in rows[1]['owners_1433_1444_1445'])
def first_bracket(result,root):
 assert result['status']=='first_boundary' and not result['accepted']
 low=Q(result['lower_time_exact']);high=Q(result['upper_time_exact'])
 if isinstance(root,Q):assert low<root<=high
 else:assert d(low)<root<=d(high)
 assert high-low<=a.TIME_WIDTH and result['total_proof_nodes']<=a.CAP
 assert result['no_earlier_root_exact'] and result['first_root_in_bracket_exact'] and result['bracket_roots']>=1
 return float(high-low)
def verify_passive(r,x,v,m,pins):
 assert r['accepted'] and r['status']=='passive'
 out=r['velocities'];delta=exact_kinetic(out,m)-exact_kinetic(v,m)
 assert delta==Q(r['represented_kinetic_change_exact_J']) and delta<=0
 P=tuple(sum((fq(m[i])*(fq(out[i][k])-fq(v[i][k])) for i in range(3)),Q(0)) for k in range(3))
 L=tuple(sum((xprod(tuple(fq(z) for z in x[i]),tuple(fq(m[i])*(fq(out[i][j])-fq(v[i][j])) for j in range(3)))[k] for i in range(3)),Q(0)) for k in range(3))
 reactionP=tuple(map(Q,r['constraint_impulse_on_anchors_exact']));reactionL=tuple(map(Q,r['constraint_angular_impulse_on_anchors_exact']))
 residualP=tuple(P[k]+reactionP[k] for k in range(3));residualL=tuple(L[k]+reactionL[k] for k in range(3))
 assert list(map(str,residualP))==r['represented_linear_balance_residual_exact']
 assert list(map(str,residualL))==r['represented_angular_balance_residual_exact']
 assert all(abs(z)<=Q(r['represented_linear_balance_tolerance_exact']) for z in residualP)
 assert all(abs(z)<=Q(r['represented_angular_balance_tolerance_exact']) for z in residualL)
 assert all(norm(speed)<30 for speed in tuple(v)+tuple(out))
 assert all(out[i]==tuple(float(z) for z in v[i]) for i in pins)
 assert list(map(str,(-z for z in reactionP)))==r['external_support_impulse_on_free_body_exact']
 return float(delta),max(map(lambda z:float(abs(z)),residualP)),max(map(lambda z:float(abs(z)),residualL))
def main():
 checked_before=verify_inputs();rng=random.Random(0xA2EA2813);rows=[];max_width=0.;polynomial_checks=0
 for i in range(64):
  L=rng.choice((.125,.25,.5,1.,2.,4.));h0=rng.choice((.125,.25,.5,1.,2.));h1=rng.choice((.125,.25,.5,1.,2.))
  offset=tuple(rng.randrange(-8,9)/8 for _ in range(3))
  start=(offset,(offset[0]+L,offset[1],offset[2]),(offset[0],offset[1]+h0,offset[2]))
  end=(start[0],start[1],(offset[0],offset[1]-h1,offset[2]))
  m=a.Motion(start,end);r=a.cast(m)
  # Independent planar determinant root of L*(h0-(h0+h1)t)=2G.
  # The dyadic coordinates make both offsets and differences exact.
  root=(fq(h0)-2*QG/fq(L))/(fq(h0)+fq(h1));max_width=max(max_width,first_bracket(r,root))
  assert r['root_count']==2
  p=tuple(map(Q,r['polynomial_exact']))
  for t in (Q(0),Q(1,7),root,Q(1,2),Q(1)):
   owners=tuple(tuple(fq(z)+t*(fq(w)-fq(z)) for z,w in zip(x,y)) for x,y in zip(start,end))
   c=xprod(difference(owners[1],owners[0]),difference(owners[2],owners[0]))
   determinant=sum((z*z for z in c),Q(0))-4*QG**2
   # Independent direct affine owner evaluation, no coefficient builder.
   polynomial_value=sum((coefficient*t**j for j,coefficient in enumerate(p)),Q(0))
   assert determinant==polynomial_value;polynomial_checks+=1
  rows.append({'name':f'planar-flip-{i}','analytic_first_root_exact':str(root),'cast':r})
  if i<16:
   masses=(5e-5,1e-4,2e-4);pr=a.preventive(m,masses,(0,1));assert pr['accepted'] and pr['status']=='admitted_response'
   velocities=tuple(tuple(float((fq(w)-fq(z))/fq(m.duration)) for z,w in zip(x,y)) for x,y in zip(start,end))
   verify_passive(pr['response'],pr['boundary_positions'],velocities,masses,(0,1))
   assert pr['post_response_cast']['strict_polynomial_throughout'] and pr['total_proof_nodes']<=128
   assert exact_area_squared(pr['endpoint'])>QG**2 and a.represented_area(pr['endpoint'])>a.GATE
   rows.append({'name':f'anchored-response-{i}','result':pr})
  clear_end=(start[0],start[1],(offset[0],offset[1]+h0/2,offset[2]));clear=a.preventive(a.Motion(start,clear_end),(5e-5,1e-4,2e-4))
  assert clear['accepted'] and clear['status']=='clear';rows.append({'name':f'clear-contraction-{i}','result':clear})
 for i in range(64):
  start=((0,0,0),(1,0,0),(0,1,0))
  end=tuple(tuple(z+rng.uniform(-.2,.2) for z in owner) for owner in start)
  motion=a.Motion(start,end);p=independent_polynomial(motion);assert independent_bernstein_positive(p)
  r=a.preventive(motion,(5e-5,1e-4,2e-4));assert r['accepted'] and r['status']=='clear'
  candidate=tuple(map(Q,r['original_cast']['polynomial_exact']))
  while len(p)>1 and p[-1]==0:p=p[:-1]
  assert p==candidate;polynomial_checks+=1
  rows.append({'name':f'general-3d-independent-bernstein-clear-{i}','result':r,'independent_bernstein_positive':True})
 unit=((0.,0.,0.),(1.,0.,0.),(0.,1.,0.))
 flip=a.Motion(unit,((0,0,0),(1,0,0),(0,-1,0)))
 free=a.preventive(flip,(1,2,3));assert not free['accepted'] and free['response']['accepted'] and free['post_response_cast']['status']=='first_boundary'
 rows.append({'name':'passive-local-rate-does-not-certify-future','result':free})
 equilateral=((0,0,0),(1,1,0),(1,0,1));eq=a.Motion(equilateral,tuple(tuple(-z for z in p) for p in equilateral));eq_cast=a.cast(eq)
 with localcontext() as ctx:
  ctx.prec=100;eq_root=(1-(2*d(QG)/Decimal(3).sqrt()).sqrt())/2;max_width=max(max_width,first_bracket(eq_cast,eq_root))
 eq_response=a.preventive(eq,(1,1,1));assert eq_response['accepted'];rows.append({'name':'free-equilateral-contraction','analytic_first_root_decimal100':str(eq_root),'result':eq_response})
 verify_passive(eq_response['response'],eq_response['boundary_positions'],tuple(tuple(-2*z for z in p) for p in equilateral),(1,1,1),())
 quartic=a.Motion(unit,((0,0,0),(-1,0,0),(0,-3,0)));qr=a.cast(quartic)
 with localcontext() as ctx:
  ctx.prec=100;root=(6-(Decimal(4)+64*d(QG)).sqrt())/16;max_width=max(max_width,first_bracket(qr,root))
 assert qr['root_count']==4;rows.append({'name':'four-roots-first-event','analytic_first_root_decimal100':str(root),'cast':qr})
 touch=a.Motion(((0,0,0),(-.5,1,0),(-2*a.GATE,-.5,0)),((0,0,0),(.5,1,0),(-2*a.GATE,.5,0)))
 tr=a.cast(touch);max_width=max(max_width,first_bracket(tr,Q(1,2)));assert tr['root_count']==1
 tangent=a.preventive(touch,(1,1,1));assert not tangent['accepted'] and tangent['total_proof_nodes']<=128
 rows.append({'name':'exact-double-root-represented-strict-margin-frontier','cast':tr,'result':tangent})
 height=3*a.GATE;near=a.Motion(((0,0,0),(1,0,0),(0,height,0)),((0,0,0),(1,0,0),(0,-height,0)));nr=a.cast(near)
 root=(fq(height)-2*QG)/(2*fq(height));max_width=max(max_width,first_bracket(nr,root));rows.append({'name':'near-inverted-crossing','cast':nr,'analytic_first_root_exact':str(root)})
 terminal=a.cast(a.Motion(unit,((0,0,0),(1,0,0),(0,2*a.GATE,0))));max_width=max(max_width,first_bracket(terminal,Q(1)));rows.append({'name':'strict-terminal-equality','cast':terminal})
 rest,failed=fixtures();chord=a.Motion(rest,failed);cr=a.preventive(chord,(5e-5,)*3);assert cr['accepted'] and cr['status']=='admitted_response'
 assert a.represented_area(failed)<a.GATE and exact_area_squared(cr['endpoint'])>QG**2
 velocities=tuple(tuple(float(fq(w)-fq(z)) for z,w in zip(x,y)) for x,y in zip(rest,failed));verify_passive(cr['response'],cr['boundary_positions'],velocities,(5e-5,)*3,())
 rows.append({'name':'declared-synthetic-rest-to-frame620-chord-not-actual-history','result':cr})
 # Domain and singularity controls must reject before any geometry clear.
 domain=[]
 for label,masses,pins in (
  ('negative-mass',(-1,1,1),()),('zero-mass',(0,1,1),()),('nan-mass',(math.nan,1,1),()),('infinite-mass',(math.inf,1,1),()),('wrong-mass-count',(1,1),()),
  ('bad-pin-high',(1,1,1),(3,)),('bad-pin-negative',(1,1,1),(-1,)),('duplicate-pin',(1,1,1),(0,0)),('float-pin',(1,1,1),(0.,)),('boolean-pin',(1,1,1),(True,))):
  r=a.preventive(a.Motion(unit,unit),masses,pins);assert not r['accepted'] and r['total_proof_nodes']==0;domain.append({'name':label,'result':r})
 moving=a.preventive(a.Motion(unit,((.1,.1,0),unit[1],unit[2])),(1,1,1),(0,));assert moving['status']=='unsupported_moving_pin';domain.append({'name':'moving-pin-even-on-geometrically-clear-path','result':moving})
 underflow_pin=a.preventive(a.Motion(unit,((math.ulp(0.),0,0),unit[1],unit[2]),1e308),(1,1,1),(0,));assert underflow_pin['status']=='unsupported_moving_pin';domain.append({'name':'moving-pin-derived-velocity-underflow','result':underflow_pin})
 for label,face in (('zero-area',((0,0,0),(1,0,0),(2,0,0))),('below-area-gate',((0,0,0),(1,0,0),(0,a.GATE,0))),('initial-equality',((0,0,0),(1,0,0),(0,2*a.GATE,0))),('underflow-area',((0,0,0),(1e-200,0,0),(0,1e-200,0)))):
  r=a.cast(a.Motion(face,face));assert not r['accepted'] and r['status']=='initial_area_violation';domain.append({'name':label,'result':r})
 no_free=a.passive_velocity(unit,((0,0,0),)*3,(1,1,1),(0,1,2));assert not no_free['accepted'] and no_free['status']=='unresolved';domain.append({'name':'zero-free-gradient-response','result':no_free})
 speed_end=tuple((p[0]+30,p[1],p[2]) for p in unit);speed=a.preventive(a.Motion(unit,speed_end),(1,1,1));assert speed['status']=='scene_speed_failure';domain.append({'name':'strict-speed-equality','result':speed})
 slow=a.preventive(a.Motion(unit,tuple((p[0]+29,p[1],p[2]) for p in unit)),(1,1,1));assert slow['accepted'];domain.append({'name':'within-speed-domain-clear','result':slow})
 for label,budget in (('exhausted-budget',a.Budget(128)),('invalid-negative-budget',a.Budget(-1))):
  r=a.cast(flip,budget);assert not r['accepted'];domain.append({'name':label,'result':r})
 mutable=[[0,0,0],[1,0,0],[0,1,0]];private=a.Motion(mutable,mutable);mutable[2][1]=0;assert private.start[2][1]==1 and a.cast(private)['accepted'];domain.append({'name':'private-immutable-owner-copy','status':'pass'})
 for label,start,end,duration in (('nan-owner',((math.nan,0,0),unit[1],unit[2]),unit,1),('infinite-owner',((math.inf,0,0),unit[1],unit[2]),unit,1),('zero-duration',unit,unit,0),('nan-duration',unit,unit,math.nan)):
  try:a.Motion(start,end,duration)
  except a.Invalid:domain.append({'name':label,'status':'invalid'});continue
  raise AssertionError(label)
 # Exact actual rounded energy and represented balance controls.
 passive=[];maxP=0.;maxL=0.;maxK=0.;passivity_rejections=0
 for i in range(128):
  face=tuple(tuple(rng.uniform(-1,1) for k in range(3)) for j in range(3));vel=tuple(tuple(rng.uniform(-4,4) for k in range(3)) for j in range(3));masses=tuple(rng.choice((5e-5,1e-4,2e-4)) for j in range(3));pins=(0,) if i%4==0 else ()
  if pins:vel=((0.,0.,0.),vel[1],vel[2])
  r=a.passive_velocity(face,vel,masses,pins)
  if not r['accepted']:
   assert r['status']=='represented_passivity_failure';passivity_rejections+=1
  else:
   delta,p,l=verify_passive(r,face,vel,masses,pins);maxP=max(maxP,p);maxL=max(maxL,l);maxK=max(maxK,delta)
  passive.append({'name':f'ordinary-represented-passive-{i}','owners':face,'velocities':vel,'masses':masses,'pins':pins,'response':r})
 sealed=old_study();counterexamples={}
 for direction in ('y','x'):
  common=(0,1e12,0) if direction=='y' else (1e12,0,0);vel=tuple(tuple(z+(1 if i==0 and k<2 else 0) for k,z in enumerate(common)) for i in range(3));masses=(1,2,3)
  old=sealed.reciprocal_impulse(unit,vel,masses);old_delta=exact_kinetic(old['velocities'],masses)-exact_kinetic(vel,masses);new=a.passive_velocity(unit,vel,masses)
  assert old['status']=='projected' and not new['accepted']
  if direction=='y':assert old_delta>0 and new['status']=='represented_passivity_failure'
  else:assert new['status']=='represented_reciprocity_failure' and Q(new['represented_kinetic_change_exact_J'])<0
  counterexamples[f'high-common-{direction}']={'old_sealed_status':old['status'],'old_exact_actual_kinetic_change_J':str(old_delta),'old_float_kinetic_change_J':old['kinetic_after_J']-old['kinetic_before_J'],'new':new}
 pin_vel=((1,1,0),(0,0,0),(0,0,0));old=sealed.reciprocal_impulse(unit,pin_vel,(1,1,1),(0,));new=a.passive_velocity(unit,pin_vel,(1,1,1),(0,))
 assert old['status']=='projected' and exact_kinetic(old['velocities'],(1,1,1))-exact_kinetic(pin_vel,(1,1,1))==1 and new['status']=='unsupported_moving_pin'
 counterexamples['moving-pin']={'old_sealed_status':old['status'],'old_exact_actual_kinetic_gain_J':'1','new':new}
 # Retain original position-derived work without changing the sealed study.
 first_result=json.loads((STUDY/'result.json').read_text());projection=first_result['position_projections'][0] if 'position_projections' in first_result else None
 if projection is None:
  projection=next(v[0] for k,v in first_result.items() if isinstance(v,list) and v and isinstance(v[0],dict) and 'reconstructed_velocity' in v[0])
 kinetic_gain=projection['reconstructed_velocity'][0]['injected_kinetic_from_declared_rest_J'];assert kinetic_gain>0
 counterexamples['position-reconstruction']={'declared_zero_velocity_kinetic_gain_J':kinetic_gain,'sealed_result_sha256':hashlib.sha256((STUDY/'result.json').read_bytes()).hexdigest(),'no_preventive_position_projection':True}
 temporal_nodes=[r.get('result',r.get('cast',{})).get('total_proof_nodes',0) for r in rows]
 all_nodes=temporal_nodes+[r.get('result',{}).get('total_proof_nodes',0) for r in domain]
 assert max(all_nodes)<=128
 result={'scope':'bounded CPU FP64 represented-owner exact-rational polynomial/velocity primitive; no full-drop repair',
  'frozen_binding_checks_before':checked_before,'frozen_binding_checks_after':verify_inputs(),'area_gate_strict_m2':a.GATE,'strict_scene_speed_limit_m_s':a.SPEED_LIMIT,
  'temporal_proof_node_cap':a.CAP,'requested_normalized_time_width_exact':str(a.TIME_WIDTH),'represented_boundary_retry_cap':16,
  'analytic_planar_first_root_cases':64,'anchored_response_cases':16,'safe_linear_contraction_cases':64,'general_3d_independent_bernstein_clear_cases':64,'direct_polynomial_identity_checks':polynomial_checks,
  'ordinary_represented_velocity_cases':128,'ordinary_velocity_admitted':128-passivity_rejections,'ordinary_velocity_positive_energy_rejected':passivity_rejections,
  'maximum_actual_represented_linear_balance_residual_kg_m_s':maxP,'maximum_actual_represented_angular_balance_residual_kg_m2_s':maxL,'maximum_admitted_kinetic_change_J':maxK,
  'maximum_first_root_bracket_width':max_width,'maximum_total_proof_nodes':max(all_nodes),'maximum_temporal_proof_nodes':max(temporal_nodes),'named_temporal_cases':len(rows),'domain_cases':len(domain),
  'source_owned_chord_is_synthetic':True,'retained_full_CPU96_fail_unchanged':True,'temporal_cases':rows,'domain_controls':domain,'passive_controls':passive,'rejected_mechanisms':counterexamples}
 return result

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--negative',choices=('moving-pin','high-common-y','high-common-x','position-reconstruction','local-rate-only','tangent-margin'));args=parser.parse_args()
 result=main()
 if args.negative:
  if args.negative in result['rejected_mechanisms']:control=result['rejected_mechanisms'][args.negative]
  else:
   name='passive-local-rate-does-not-certify-future' if args.negative=='local-rate-only' else 'exact-double-root-represented-strict-margin-frontier'
   control=next(row for row in result['temporal_cases'] if row['name']==name)
  print(json.dumps({'qualification':'REJECTED_OR_UNRESOLVED','negative_control':args.negative,'retained_control':control},sort_keys=True));sys.exit(1)
 (HERE/'result.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
 summary={key:value for key,value in result.items() if key not in ('temporal_cases','domain_controls','passive_controls','rejected_mechanisms')}
 print(json.dumps({'qualification':'PASS','metrics':summary},sort_keys=True))
