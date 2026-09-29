#!/usr/bin/env python3
"""Bounded CPU-only local material-area mechanics study; no trajectory repair."""
import argparse,copy,hashlib,json,math,random,sys
from decimal import Decimal,localcontext
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent.parent
GATE=1e-8;DT=1/120/96;MASS=5e-5;SOURCE_COMPLIANCE=1e-8

def add(a,b):return tuple(x+y for x,y in zip(a,b))
def sub(a,b):return tuple(x-y for x,y in zip(a,b))
def mul(a,s):return tuple(x*s for x in a)
def dot(a,b):return math.fsum(x*y for x,y in zip(a,b))
def cross(a,b):return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])
def norm(a):return math.hypot(*a)
def vsum(a):return tuple(math.fsum(x[k] for x in a) for k in range(3))
def area(x):return .5*norm(cross(sub(x[1],x[0]),sub(x[2],x[0])))
def gradient(x,normal=None):
 c=cross(sub(x[1],x[0]),sub(x[2],x[0]));l=norm(c)
 if normal is None:
  if not l>0 or not math.isfinite(l):raise ValueError('degenerate face: no unsigned-area normal')
  normal=mul(c,1/l)
 return [mul(cross(normal,sub(x[2],x[1])),.5),mul(cross(normal,sub(x[0],x[2])),.5),mul(cross(normal,sub(x[1],x[0])),.5)]
def decimal_area(x):
 p=[[Decimal.from_float(float(z)) for z in q] for q in x];e=[[p[i][k]-p[0][k] for k in range(3)] for i in (1,2)]
 c=[e[0][1]*e[1][2]-e[0][2]*e[1][1],e[0][2]*e[1][0]-e[0][0]*e[1][2],e[0][0]*e[1][1]-e[0][1]*e[1][0]]
 return sum(z*z for z in c).sqrt()/2

def decimal_gradient(x):
 # Independent high-precision finite difference of the scalar norm area.
 p=[[Decimal.from_float(float(z)) for z in q] for q in x]
 h=Decimal('1e-20');result=[]
 def measure(p):
  e=[[p[i][k]-p[0][k] for k in range(3)] for i in (1,2)]
  c=[e[0][1]*e[1][2]-e[0][2]*e[1][1],e[0][2]*e[1][0]-e[0][0]*e[1][2],e[0][0]*e[1][1]-e[0][1]*e[1][0]]
  return sum(z*z for z in c).sqrt()/2
 for i in range(3):
  row=[]
  for k in range(3):
   a=copy.deepcopy(p);b=copy.deepcopy(p);a[i][k]+=h;b[i][k]-=h
   row.append(float((measure(a)-measure(b))/(2*h)))
  result.append(tuple(row))
 return result

def kinetic(v,m):return math.fsum(.5*mass*dot(speed,speed) for mass,speed in zip(m,v))
def momentum(v,m):return vsum(mul(p,z) for p,z in zip(v,m))
def angular(x,v,m):return vsum(cross(p,mul(q,z)) for p,q,z in zip(x,v,m))
def reciprocal_impulse(x,v,m,pinned=()):
 g=gradient(x);w=[0 if i in pinned else 1/m[i] for i in range(3)];W=math.fsum(wi*dot(gi,gi) for wi,gi in zip(w,g));rate=math.fsum(dot(a,b) for a,b in zip(g,v))
 if not W>0 or not math.isfinite(W):return {'status':'unresolved','reason':'no positive finite mass metric'}
 p=max(0.,-rate/W);out=[add(vi,mul(gi,wi*p)) for vi,gi,wi in zip(v,g,w)]
 before=kinetic(v,m);after=kinetic(out,m);reaction=vsum(mul(g[i],p) for i in pinned)
 reaction_torque=vsum(cross(x[i],mul(g[i],p)) for i in pinned)
 # Fixed pins have zero velocity; moving pins require separate support work.
 predicted_loss=.5*min(0.,rate)**2/W
 return {'status':'projected','velocities':out,'impulse':p,'rate_before':rate,'rate_after':math.fsum(dot(a,b) for a,b in zip(g,out)),
         'kinetic_before_J':before,'kinetic_after_J':after,'predicted_kinetic_loss_J':predicted_loss,'energy_identity_residual_J':after-before+predicted_loss,
         'momentum_balance_residual':norm(add(sub(momentum(out,m),momentum(v,m)),reaction)),
         'angular_balance_residual':norm(add(sub(angular(x,out,m),angular(x,v,m)),reaction_torque)),
         'fixed_pin_external_reaction_impulse':reaction,'fixed_pin_external_reaction_angular_impulse':reaction_torque}

def project_position(x,m,target,maximum=16):
 out=list(x);calls=0
 if not target>0:return {'status':'invalid'}
 for iteration in range(maximum):
  A=area(out)
  if A>=target:return {'status':'projected','positions':out,'area_m2':A,'outer_iterations':iteration,'scalar_trials':calls}
  try:g=gradient(out)
  except ValueError:return {'status':'degenerate','outer_iterations':iteration,'scalar_trials':calls}
  n=mul(cross(sub(out[1],out[0]),sub(out[2],out[0])),1/(2*A));W=math.fsum(dot(gi,gi)/mi for gi,mi in zip(g,m))
  if not W>0 or not math.isfinite(W):return {'status':'unresolved'}
  # Explicit 1e-5 relative overshoot gives the study's stricter target a margin;
  # it is not a calibrated material response or a changed acceptance gate.
  scale=(target-A)/W*1.00001
  for trial in range(16):
   calls+=1;candidate=[add(p,mul(gi,scale/mi)) for p,gi,mi in zip(out,g,m)]
   c=cross(sub(candidate[1],candidate[0]),sub(candidate[2],candidate[0]))
   if area(candidate)>A and dot(c,n)>0:out=candidate;break
   scale*=.5
  else:return {'status':'unresolved','reason':'bounded position line search exhausted','outer_iterations':iteration,'scalar_trials':calls}
 return {'status':'unresolved','reason':'16 outer position iterations exhausted','outer_iterations':maximum,'scalar_trials':calls}

def verify_inputs():
 bindings=json.loads((HERE/'input-bindings.json').read_text())
 for b in bindings:
  assert hashlib.sha256((ROOT/b['original']).read_bytes()).hexdigest()==b['sha256']
  if b['frozen']:assert hashlib.sha256((ROOT/b['frozen']).read_bytes()).hexdigest()==b['sha256']
 return len(bindings)
def obj(name):return [tuple(map(float,l.split()[1:])) for l in (HERE/'frozen'/name).read_text().splitlines() if l.startswith('v ')]
def fixtures():
 rest=obj('frame0.obj');failed=obj('frame620.obj');ids=[1433,1444,1445];quad=[1433,1434,1444,1445]
 return [rest[i] for i in ids],[failed[i] for i in ids],rest,failed,ids,quad

def run():
 bindings=verify_inputs();rest,x,all_rest,all_failed,ids,quad=fixtures();m=[MASS]*3;A0=area(rest);A=area(x)
 assert A<GATE and A0>GATE
 g=gradient(x)
 with localcontext() as c:
  c.prec=80;independent_A=float(decimal_area(x));oracle_g=decimal_gradient(x)
 assert abs(A-independent_A)<1e-17
 max_gradient=0.;max_translation=0.;max_torque=0.;max_energy=0.;max_p=0.;max_l=0.;max_rate=0.;count=0
 rng=random.Random(0xA2EA2813)
 faces=[x,rest,[(0,0,0),(1,0,0),(0,1,0)],[(0,0,0),(1,0,0),(.3,1e-9,0)]]
 for i in range(60):
  faces.append([tuple(rng.uniform(-1,1) for _ in range(3)) for j in range(3)])
 for face in faces:
  gradients=gradient(face)
  with localcontext() as c:
   c.prec=80;reference=decimal_gradient(face)
  max_gradient=max(max_gradient,max(norm(sub(a,b)) for a,b in zip(gradients,reference)))
  max_translation=max(max_translation,norm(vsum(gradients)));max_torque=max(max_torque,norm(vsum(cross(p,gi) for p,gi in zip(face,gradients))))
  for variant in range(4):
   masses=[10**rng.uniform(-5,-2) for _ in range(3)]
   velocities=[tuple(rng.uniform(-5,5) for _ in range(3)) for j in range(3)]
   pins=() if variant<3 else (0,)
   if pins:velocities[0]=(0,0,0)
   result=reciprocal_impulse(face,velocities,masses,pins);assert result['status']=='projected'
   assert result['kinetic_after_J']<=result['kinetic_before_J']+1e-14
   assert result['rate_after']>=-1e-12
   max_energy=max(max_energy,abs(result['energy_identity_residual_J']));max_p=max(max_p,result['momentum_balance_residual']);max_l=max(max_l,result['angular_balance_residual']);max_rate=max(max_rate,max(0.,-result['rate_after']));count+=1
 assert max_gradient<1e-12 and max_translation<1e-14 and max_torque<1e-14
 assert max_energy<1e-12 and max_p<1e-12 and max_l<1e-12
 targets=[2e-8,.001*A0,.01*A0,.1*A0];projections=[]
 for target in targets:
  r=project_position(x,m,target);assert r['status']=='projected' and r['area_m2']>GATE
  delta=[sub(b,a) for a,b in zip(x,r['positions'])];reconstruction=[]
  for dt in (DT,DT/2,DT/4,DT/16):
   speed=[mul(d,1/dt) for d in delta];reconstruction.append({'dt_s':dt,'injected_kinetic_from_declared_rest_J':kinetic(speed,m),'maximum_added_speed_m_s':max(norm(v) for v in speed)})
  assert abs(reconstruction[1]['injected_kinetic_from_declared_rest_J']/reconstruction[0]['injected_kinetic_from_declared_rest_J']-4)<1e-12
  patch=list(all_failed);patch_rest=list(all_rest)
  for index,p in zip(ids,r['positions']):patch[index]=p
  edges=[(1433,1434),(1433,1444),(1434,1445),(1444,1445)]
  def yarn_energy(points):return math.fsum((math.dist(points[a],points[b])-math.dist(patch_rest[a],patch_rest[b]))**2/(2*SOURCE_COMPLIANCE) for a,b in edges)
  r.update(target_area_m2=target,target_ratio=target/A0,maximum_owner_correction_m=max(norm(v) for v in delta),
           mass_weighted_position_sum_residual=norm(vsum(mul(d,z) for d,z in zip(delta,m))),
           old_position_gradient_torque_residual=norm(vsum(cross(p,mul(d,z)) for p,d,z in zip(x,delta,m))),
           reconstructed_velocity=reconstruction,four_existing_yarn_spring_energy_before_J=yarn_energy(all_failed),four_existing_yarn_spring_energy_after_J=yarn_energy(patch),
           neighboring_render_triangle_area_m2=area([patch[i] for i in (1433,1445,1434)]))
  declared_old_v=[(1,0,0),(0,0,0),(0,0,0)]
  r['angular_change_if_old_velocity_unchanged_kg_m2_s']=norm(sub(angular(r['positions'],declared_old_v,m),angular(x,declared_old_v,m)))
  r['four_yarn_plus_reconstructed_kinetic_change_at_native_dt_J']=yarn_energy(patch)-yarn_energy(all_failed)+reconstruction[0]['injected_kinetic_from_declared_rest_J']
  projections.append(r)
 inward=[mul(v,-1/max(norm(v),1e-30)) for v in g];impulse=reciprocal_impulse(x,inward,m)
 assert impulse['kinetic_after_J']<=impulse['kinetic_before_J'] and impulse['rate_after']>=-1e-14
 # Velocity arrest does not repair an already-invalid geometric state.
 assert A<GATE
 # Any fixed reference normal reacts to a pure rigid rotation of a healthy face.
 healthy=[(0,0,0),(1,0,0),(0,1,0)];rotated=[(p[0],-p[2],p[1]) for p in healthy];signed=.5*dot((0,0,1),cross(sub(rotated[1],rotated[0]),sub(rotated[2],rotated[0])))
 fixed_g=gradient(rotated,(0,0,1));fixed_torque=norm(vsum(cross(p,gi) for p,gi in zip(rotated,fixed_g)))
 assert area(rotated)==.5 and signed==0 and fixed_torque>.4
 # Squared area has a vanishing gradient; one raw Newton step grows as1/A.
 squared=[]
 for height in (1e-5,1e-9,1e-13):
  face=[(0,0,0),(1,0,0),(.3,height,0)];a=area(face);gA=gradient(face);gB=[mul(z,2*a) for z in gA];W=math.fsum(dot(z,z)/MASS for z in gB);dl=(1e-8-a*a)/W
  correction=[mul(z,dl/MASS) for z in gB];squared.append({'height_m':height,'area_m2':a,'maximum_raw_squared_area_newton_correction_m':max(norm(z) for z in correction)})
 assert squared[-1]['maximum_raw_squared_area_newton_correction_m']>1e4
 # Aggregate cell area cannot certify each material face.
 neighboring=area([all_failed[i] for i in (1433,1445,1434)]);aggregate=A+neighboring
 assert aggregate>GATE and A<GATE
 # End-state unsigned area checks cannot certify the intervening path.
 end=[(0,0,0),(1,0,0),(0,-1,0)];middle=[mul(add(a,b),.5) for a,b in zip(healthy,end)]
 assert area(healthy)>GATE and area(end)>GATE and area(middle)==0
 assert project_position([(0,0,0),(1,0,0),(2,0,0)],m,2e-8)['status']=='degenerate'
 assert reciprocal_impulse(x,[(0,0,0)]*3,m,(0,1,2))['status']=='unresolved'
 # Finite compliant floor remains finite-energy but may violate a hard gate.
 J=A/A0;Jmin=2e-8/A0;gJ=[mul(z,1/A0) for z in g];W=math.fsum(dot(z,z)/MASS for z in gJ);alpha=1.0;lambda_pos=(Jmin-J)/(W+alpha/DT**2)
 compliant=[add(p,mul(gi,lambda_pos/MASS)) for p,gi in zip(x,gJ)]
 barrier_before=.5*(J-Jmin)**2/alpha;barrier_after=.5*min(0.,area(compliant)/A0-Jmin)**2/alpha
 soft_gate_dt=DT/1000
 soft_lambda=(Jmin-J)/(W+alpha/soft_gate_dt**2)
 soft_small=[add(p,mul(gi,soft_lambda/MASS)) for p,gi in zip(x,gJ)]
 assert math.isfinite(barrier_before) and area(compliant)<2e-8 and area(soft_small)<GATE
 result={'scope':'bounded local CPU math; saved cloth velocities absent, all velocity cases declared synthetic; no full-drop repair',
         'bindings_verified_before':bindings,'unchanged_strict_area_gate_m2':GATE,'retained_triangle':2813,'owners':ids,'authored_owner_masses_kg':m,
         'rest_area_m2':A0,'retained_area_m2':A,'decimal80_retained_area_m2':independent_A,'area_ratio':A/A0,'gate_ratio_to_rest':GATE/A0,
         'retained_gradients':g,'independent_decimal80_gradients':oracle_g,
         'gradient_faces':len(faces),'reciprocal_velocity_cases':count,'maximum_gradient_oracle_error_m':max_gradient,
         'maximum_gradient_translation_residual_m':max_translation,'maximum_gradient_torque_residual_m2':max_torque,
         'maximum_velocity_energy_identity_residual_J':max_energy,'maximum_impulse_momentum_balance_residual_kg_m_s':max_p,'maximum_impulse_angular_balance_residual_kg_m2_s':max_l,'maximum_remaining_closing_rate_m2_s':max_rate,
         'retained_cell_synthetic_inward_velocity_impulse':impulse,'bounded_position_repairs':projections,
         'position_outer_cap':16,'position_scalar_line_search_cap_per_outer':16,
         'rejected_mechanisms':{'aggregate_cell_area_blindspot':{'collapsed_face_m2':A,'neighbor_area_m2':neighboring,'aggregate_m2':aggregate},
          'fixed_world_normal':{'unsigned_rigidly_rotated_area_m2':area(rotated),'signed_fixed_normal_area_m2':signed,'nonzero_net_torque_per_scalar_impulse_m2':fixed_torque},
          'raw_squared_area_newton':squared,'endpoint_only':{'start_area_m2':area(healthy),'end_area_m2':area(end),'midpoint_area_m2':area(middle)},
          'velocity_only_does_not_restore_invalid_area':True,'position_velocity_reconstruction_has_dt_inverse_square_energy':True},
         'illustrative_finite_compliance':{'compliance_inverse_J':alpha,'barrier_energy_before_J':barrier_before,'barrier_energy_after_J':barrier_after,'area_after_one_soft_step_m2':area(compliant),'area_after_dt_reduced_1000_m2':area(soft_small),'hard_gate_qualification':False,'material_calibrated':False}}
 result['bindings_verified_after']=verify_inputs()
 (HERE/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 print(json.dumps({k:v for k,v in result.items() if k not in ('bounded_position_repairs','retained_cell_synthetic_inward_velocity_impulse','retained_gradients','independent_decimal80_gradients')},indent=2));return 0

def negative(mode):
 rest,x,rr,xx,ids,quad=fixtures()
 if mode=='reconstruction-energy':
  out=project_position(x,[MASS]*3,2e-8);d=[sub(b,a) for a,b in zip(x,out['positions'])];K=kinetic([mul(z,1/DT) for z in d],[MASS]*3)
  print(json.dumps({'negative':'unaccounted_position_reconstruction_work','initial_declared_K_J':0,'after_K_J':K,'area_after_m2':out['area_m2'],'energy_increase':K>0}));return 1 if K>0 else 2
 if mode=='squared-singularity':
  face=[(0,0,0),(1,0,0),(.3,1e-13,0)];A=area(face);g=[mul(z,2*A) for z in gradient(face)];W=math.fsum(dot(z,z)/MASS for z in g);dl=(1e-8-A*A)/W;delta=max(norm(mul(z,dl/MASS)) for z in g)
  print(json.dumps({'negative':'raw_squared_area_newton_singularity','initial_area_m2':A,'maximum_correction_m':delta}));return 1 if delta>1e4 else 2
 if mode=='aggregate-cell':
  A=area(x);total=A+area([xx[i] for i in (1433,1445,1434)]);print(json.dumps({'negative':'aggregate_area_hides_face_collapse','face_area_m2':A,'aggregate_area_m2':total,'false_gate_acceptance':total>GATE and A<GATE}));return 1 if total>GATE and A<GATE else 2
 raise AssertionError('mode')
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--negative',choices=('reconstruction-energy','squared-singularity','aggregate-cell'));args=parser.parse_args()
 try:exitcode=negative(args.negative) if args.negative else run()
 except Exception as e:print('FAIL',repr(e),file=sys.stderr);exitcode=2
 sys.exit(exitcode)
