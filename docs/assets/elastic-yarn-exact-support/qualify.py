#!/usr/bin/env python3
"""CPU-only qualification of exact closed support, independent of production.
Analytic roots + frozen independent FP64 active-set QP are separate oracles.
"""
import argparse, copy, ctypes, hashlib, importlib.util, json, math, random, struct, sys
from collections import Counter
from decimal import Decimal, localcontext
from fractions import Fraction as Q
from pathlib import Path
from closed_support import Motion, solve, f32, exact, Invalid, BRIDGE, TOLERANCE, OUTER_BUDGET
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
R=.004
TRI=[[-.3,-.3,0],[.3,-.3,0],[0,.3,0]]
def retained(): return Motion([[-.8,.6,R],[.8,.6,R]]+copy.deepcopy(TRI),[[-.8,-.6,R],[.8,-.6,R]]+copy.deepcopy(TRI))
def transform(m,perm=(0,1,2),signs=(1,1,1),winding=False,yarn=False,time=False):
 a=[[signs[k]*p[perm[k]] for k in range(3)] for p in m.start];b=[[signs[k]*p[perm[k]] for k in range(3)] for p in m.end]
 if winding:a[3],a[4]=a[4],a[3];b[3],b[4]=b[4],b[3]
 if yarn:a[0],a[1]=a[1],a[0];b[0],b[1]=b[1],b[0]
 if time:a,b=b,a
 return Motion(a,b,m.radius)
def double_root():
 t=[[0,0,0],[-1,1,0],[-1,2,0]]
 return Motion([[-.5,0,R],[.5,-.5,R]]+t,[[.5,0,R],[1.5,.5,R]]+t)
def quadratic(c=Q(1,8)):
 c=float(c);t=[[0,0,0],[-1,1,0],[-1,2,0]]
 return Motion([[0,-c,R],[1,-c,R]]+t,[[-1,-c,R],[0,1-c,R]]+t)
def oblique():
 r=1/128;t=[[-.5,-2,1.5],[.5,-2,1.5],[0,2,-1.5]]
 return Motion([[-1,4,-3+5*r/4],[1,4,-3+5*r/4]]+t,[[-1,-4,3+5*r/4],[1,-4,3+5*r/4]]+t,r)
def rotation():
 r=f32(R)
 return Motion([[-.25,0,r],[.25,0,r],[-.5,-.5,0],[.5,-.5,0],[0,.5,0]],
               [[-.25,-r,0],[.25,-r,0],[-.5,0,-.5],[.5,0,-.5],[0,0,.5]])
def nextf(x,up):
 bits=struct.unpack('I',struct.pack('f',f32(x)))[0]
 return struct.unpack('f',struct.pack('I',bits+(1 if up else -1)))[0]
def height(delta):
 m=retained();z=nextf(m.radius,delta>0)
 for poses in (m.start,m.end):
  for i in range(2):poses[i][2]=z
 return m

def decimal_q(q):return Decimal(q.numerator)/Decimal(q.denominator)
def independent_material(m,result):
 material=result.get('material',{})
 if 'yarn_u_exact' not in material:return
 u=Q(material['yarn_u_exact']);w=[Q(x) for x in material['triangle_weights_exact']]
 assert 0<=u<=1 and all(0<=x<=1 for x in w) and sum(w)==1,'infeasible rational material pair'
 if result['status']=='grazing':t=Q(result['root_exact'])
 else:t=Q(0)
 a,b=m.qpoints();p=[tuple(x+t*(y-x) for x,y in zip(xp,yp)) for xp,yp in zip(a,b)]
 yarn=tuple((1-u)*p[0][k]+u*p[1][k] for k in range(3));tri=tuple(sum(w[i]*p[2+i][k] for i in range(3)) for k in range(3))
 distance2=sum((x-y)**2 for x,y in zip(yarn,tri))
 assert distance2==exact(m.radius)**2,'material pair does not exactly touch the represented radius'

def controls():
 cases=[]
 def add(name,m,status='grazing',root=None):cases.append((name,m,status,root))
 base=retained()
 for perm in ((0,1,2),(1,2,0),(2,0,1)):
  for signs in ((1,1,1),(-1,1,-1),(1,-1,1)):
   for winding in (False,True):
    for yarn in (False,True):
     add(f'retained_perm{perm}_sign{signs}_w{int(winding)}_y{int(yarn)}',transform(base,perm,signs,winding,yarn),root=Q(1,4))
 add('retained_reverse_time',transform(base,time=True),root=Q(1,4))
 add('isolated_double_root',double_root(),root=Q(1,2))
 add('isolated_double_root_reversed',transform(double_root(),time=True,winding=True),root=Q(1,2))
 add('irrational_quadratic',quadratic(),root=('sqrt',Q(1,8)))
 add('irrational_quadratic_reversed_winding',transform(quadratic(),winding=True,yarn=True),root=('sqrt',Q(1,8)))
 add('oblique_projection',oblique(),root=Q(1277,5120))
 add('oblique_projection_reversed_winding',transform(oblique(),winding=True,yarn=True),root=Q(1277,5120))
 add('one_ulp_above_support',height(1),'clear')
 add('one_ulp_below_support',height(-1),'unsupported_closed_plane')
 translated=retained()
 for poses in (translated.start,translated.end):
  for owner in poses:owner[2]+=4096
 add('fresh_translated_fp32_rounding_overlap',Motion(translated.start,translated.end,translated.radius),'unsupported_closed_plane')
 t=[[0,0,0],[1,0,0],[0,1,0]]
 add('collinear_edge_entry',Motion([[-1,0,R],[-.5,0,R]]+t,[[1,0,R],[1.5,0,R]]+t),root=Q(1,4))
 add('degenerate_yarn_point_entry',Motion([[0,.6,R],[0,.6,R]]+TRI,[[0,-.6,R],[0,-.6,R]]+TRI),root=Q(1,4))
 add('vertex_extension_outside_yarn',Motion([[3,.6,R],[4,.6,R]]+TRI,[[3,-.6,R],[4,-.6,R]]+TRI),'clear')
 add('projected_clear_never_reaches',Motion([[-.8,.6,R],[.8,.6,R]]+TRI,[[-.8,.5,R],[.8,.5,R]]+TRI),'clear')
 add('initial_supported_tangent',Motion([[-.8,0,R],[.8,0,R]]+TRI,[[-.78,.05,R],[.82,.05,R]]+TRI),'support')
 r=1/128
 # Five independent in-plane owners plus exactly represented common normal translation.
 a=[[-1,.75,0+r],[1,.75,0+r],[-.25,-.25,0],[.25,-.25,0],[0,.25,0]]
 b=[[-.9375,-.75,.25+r],[1.0625,-.75,.25+r],[-.3125,-.1875,.25],[.3125,-.3125,.25],[.0625,.375,.25]]
 add('parallel_translating_deforming_support',Motion(a,b,r),root=Q(4,13))
 add('rotating_endpoint_support_trap',rotation(),'unsupported_closed_plane')
 collapse=retained();collapse.end[2],collapse.end[3]=collapse.start[3],collapse.start[2]
 add('interior_triangle_collapse',collapse,'degenerate')
 # Exact independent linear analytic root: horizontal yarn spans every triangle
 # vertex. First overlap is min of the three represented vertex-height roots.
 rng=random.Random(0x5A17C10)
 for i in range(64):
  ta=[[-.25+rng.randrange(-4,5)/128,-.25+rng.randrange(-4,5)/128,0],
      [.25+rng.randrange(-4,5)/128,-.25+rng.randrange(-4,5)/128,0],
      [rng.randrange(-4,5)/128,.25+rng.randrange(-4,5)/128,0]]
  tb=[[p[0]+rng.randrange(-4,5)/128,p[1]+rng.randrange(-4,5)/128,0] for p in ta]
  a=[[-1,.75,R],[1,.75,R]]+ta;b=[[-1+rng.randrange(-4,5)/64,-.75,R],[1+rng.randrange(-4,5)/64,-.75,R]]+tb
  m=Motion(a,b);start,end=m.qpoints();roots=[]
  for j in range(2,5):
   numerator=start[0][1]-start[j][1]
   denominator=(end[j][1]-start[j][1])-(end[0][1]-start[0][1])
   roots.append(numerator/denominator)
  add(f'seeded_deforming_linear_{i:02d}',m,root=min(roots))
 # Analytic irreducible/double rational quadratic family: orientation c-t^2,
 # material u=t at first event. c is exactly represented, no root sampling.
 for i in range(32):
  c=Q(13+11*i,1024)
  square=math.isqrt(c.numerator)*math.isqrt(c.numerator)==c.numerator and math.isqrt(c.denominator)*math.isqrt(c.denominator)==c.denominator
  expected=Q(math.isqrt(c.numerator),math.isqrt(c.denominator)) if square else ('sqrt',c)
  add(f'seeded_quadratic_{i:02d}',quadratic(c),root=expected)
 return cases

def helper_controls(m):
 checks=0
 raw=[]
 def query(n,l=0.,h=1.,input=None):
  normal=(ctypes.c_double*3)(*n);a=m.data() if input is None else input
  return BRIDGE.safe_plane_lower(a,l,h,normal)
 for n in ((math.nan,0,1),(math.inf,0,1),(-math.inf,0,1),(0,0,0),(1e-300,0,0),(1e308,0,0)):
  assert query(n)==-math.inf,('invalid normal not failclosed',n);checks+=1
 for l,h in ((math.nan,1),(0,math.nan),(-.01,1),(0,1.01),(.5,.4)):
  assert query((0,0,1),l,h)==-math.inf,('invalid time interval not failclosed',l,h);checks+=1
 for index in (0,12,17,29,30):
  for bad in (math.nan,math.inf):
   a=m.data();a[index]=bad;assert query((0,0,1),input=a)==-math.inf;checks+=1
 a=m.data();a[30]=0;assert query((0,0,1),input=a)==-math.inf;checks+=1
 valid=query((0,0,1));assert math.isfinite(valid) and valid<=0 and valid>-1e-12;checks+=1
 normal=(ctypes.c_double*3)(math.nan,0,1)
 legacy=BRIDGE.legacy_plane_lower(m.data(),0,1,normal)
 assert legacy>1e300 and query((math.nan,0,1))==-math.inf;checks+=1
 return {'checks':checks,'legacy_nan_false_separation_m':legacy,'safe_nan_separation':'-Infinity','valid_closed_plane_lower_m':valid}

def snapshot_check():
 manifest=json.loads((HERE/'frozen-manifest.json').read_text())
 for entry in manifest['files']:
  for key in ('source','frozen'):
   p=ROOT/entry[key]
   assert hashlib.sha256(p.read_bytes()).hexdigest()==entry['sha256'],f'changed immutable input: {p}'
 return len(manifest['files'])

def mutable_controls():
 checks=0
 translated=retained()
 for poses in (translated.start,translated.end):
  for owner in poses:owner[2]+=4096
 result=solve(translated)
 assert result['status']=='invalid' and not result['accepted'] and 'normal_closing_exact' not in result
 assert translated.gap(.5)<-TOLERANCE;checks+=1
 for kind,bad in (('owner',math.nan),('owner',math.inf),('owner',1e300),('radius',math.nan),('radius',math.inf),('radius',0),('radius',f32(R)+1e-12)):
  m=retained()
  if kind=='owner':m.start[0][0]=bad
  else:m.radius=bad
  result=solve(m);assert result['status']=='invalid' and not result['accepted'] and 'normal_closing_exact' not in result;checks+=1
 for malformed in (0,1):
  m=retained()
  if malformed:m.end.pop()
  else:m.start[0].pop()
  assert solve(m)['status']=='invalid';checks+=1
 # An exactly represented mutation is a valid new input, copied privately.
 m=retained()
 for poses in (m.start,m.end):
  for owner in poses:owner[0]=f32(owner[0]+2)
 result=solve(m);assert result['status']=='grazing' and result['root_exact']=='1/4';checks+=1
 # Simulate a caller mutation after entry validation but before the first
 # bridge certificate. Only the private snapshot may reach later predicates.
 original=retained();initial=Motion(original.start,original.end,original.radius)
 admit=BRIDGE.admit_face;captured=[]
 def mutate_caller(data,cap,used):
  captured.append([float(x) for x in data])
  for poses in (original.start,original.end):
   for owner in poses:owner[2]+=4096
  return admit(data,cap,used)
 try:
  BRIDGE.admit_face=mutate_caller;result=solve(original)
 finally:BRIDGE.admit_face=admit
 assert result['status']=='grazing' and result['root_exact']=='1/4'
 assert captured==[[x for p in initial.start+initial.end for x in p]+[initial.radius]]
 assert initial.gap(.5)==0 and original.gap(.5)<-TOLERANCE;checks+=1
 return {'checks':checks,'former_narrowed_gap_m':translated.gap(.5),'unrepresented_motion_status':'invalid','private_snapshot_callers_mutation_control':True}

def run():
 integrity=snapshot_check();records=[];counts=Counter();max_space=0.;max_gap=0.;max_nodes=0;analytic=0
 for name,m,expected,root in controls():
  result=solve(m);replay=solve(m)
  assert result==replay,('replay mismatch',name)
  assert result['status']==expected,(name,result,expected)
  assert result['outer_nodes']<=OUTER_BUDGET
  assert result['accepted']==(expected in ('grazing','support','clear'))
  if expected in ('unsupported_closed_plane','degenerate'):
   assert 'normal_closing_exact' not in result,'unproved normal-closing diagnostic on rejected motion'
  if expected in ('grazing','support'):
   assert result['global_nonpenetration_exact'] and result['normal_closing_exact']=='0'
   independent_material(m,result)
  if root is not None:
   with localcontext() as ctx:
    ctx.prec=100
    exact_time=decimal_q(root[1]).sqrt() if isinstance(root,tuple) else decimal_q(root)
    assert Decimal.from_float(result['lower_time'])<=exact_time<=Decimal.from_float(result['upper_time']),(name,'analytic root outside bracket',result,exact_time)
   gap=m.gap(float(exact_time));max_gap=max(max_gap,abs(gap));assert abs(gap)<2e-10,(name,'independent FP64 gap',gap)
   if result['lower_time']>0:
    before=m.gap(result['lower_time']*.5);assert before>0,(name,'prefix not clear in independent QP',before)
   max_space=max(max_space,result['temporal_spatial_width_upper_m']);assert result['gap_at_lower_upper_m']<=TOLERANCE
   assert result['contact_at_algebraic_root_exact'] and not result['represented_penetration_used'] and not result['response_impulse_claimed'];analytic+=1
  elif expected=='clear':
   for t in (0,.25,.5,.75,1):assert m.gap(t)>=-2e-12,(name,'oracle penetration on clear',t,m.gap(t))
  if name.startswith('isolated_double_root'):
   assert m.gap(.499)>0 and m.gap(.501)>0,'double root was not isolated tangency'
  if name=='one_ulp_below_support':assert m.gap(.5)<0,'represented penetration control absent'
  if name=='rotating_endpoint_support_trap':assert abs(m.gap(0))<1e-12 and abs(m.gap(1))<1e-12 and m.gap(.5)<-TOLERANCE,'rotating trap absent'
  counts[expected]+=1;max_nodes=max(max_nodes,result['outer_nodes'])
  record={'name':name,'start':m.start,'end':m.end,'radius':m.radius,'expected':expected,'analytic_root':str(root),'result':result,'oracle_gap_at_bracket_midpoint_m':m.gap((result['lower_time']+result['upper_time'])*.5)}
  records.append(record)
  print(json.dumps({'case':name,'status':expected,'lower_time':result['lower_time'],'upper_time':result['upper_time'],'outer_nodes':result['outer_nodes'],'oracle_gap_at_midpoint_m':record['oracle_gap_at_bracket_midpoint_m']},sort_keys=True))
 helpers=helper_controls(retained())
 mutations=mutable_controls()
 for cap in (0,1,9):
  result=solve(retained(),cap);assert result['status']=='unresolved' and not result['accepted'] and result['outer_nodes']<=cap
 for cap in (-1,129,1.5):assert solve(retained(),cap)['status']=='invalid'
 constructor_invalid=0
 for radius in (0,-1,math.inf,math.nan):
  try:Motion(retained().start,retained().end,radius)
  except Invalid:constructor_invalid+=1
  else:raise AssertionError('invalid radius admitted')
 for bad in (math.nan,math.inf,1e300):
  a=retained().start;a[1][2]=bad
  try:Motion(a,retained().end)
  except Invalid:constructor_invalid+=1
  else:raise AssertionError('invalid owner admitted')
 assert constructor_invalid==7
 summary={'actual_exit_expected':0,'cases':len(records),'statuses':dict(counts),'analytic_root_controls':analytic,'maximum_temporal_spatial_width_upper_m':max_space,
          'maximum_abs_independent_oracle_gap_at_analytic_root_m':max_gap,'maximum_outer_nodes':max_nodes,'outer_cap':OUTER_BUDGET,'tolerance_m':TOLERANCE,
          'exact_replays_per_case':2,'helper_controls':helpers,'budget_failure_controls':6,'constructor_invalid_controls':constructor_invalid,
          'mutable_motion_binding_controls':mutations,
          'immutable_original_and_frozen_files_verified':integrity,'represented_penetration_admissions':0,'response_impulses_claimed':0,'no_gpu':True,'no_trajectories':True}
 (HERE/'controls.json').write_text(json.dumps({'summary':summary,'cases':records},indent=2,allow_nan=False)+'\n')
 print('SUMMARY '+json.dumps(summary,sort_keys=True));return 0

def negative(mode):
 if mode=='nan-plane':
  m=retained();n=(ctypes.c_double*3)(math.nan,0,1);raw=BRIDGE.legacy_plane_lower(m.data(),0.,1.,n);safe=BRIDGE.safe_plane_lower(m.data(),0.,1.,n)
  failed=raw>0 and safe==-math.inf;print(json.dumps({'negative':'legacy_nonfinite_normal','legacy_clearance_m':raw,'safe_failclosed':safe==-math.inf,'rejected_classifier':failed}));return 1 if failed else 2
 if mode=='rounded-penetration':
  m=height(-1);gap=m.gap(.5);rounded_accept=abs(gap)<TOLERANCE;result=solve(m)
  failed=rounded_accept and gap<0 and not result['accepted'];print(json.dumps({'negative':'round_shallow_overlap_to_support','represented_gap_m':gap,'tolerance_rounding_would_admit':rounded_accept,'exact_status':result['status'],'rejected_classifier':failed}));return 1 if failed else 2
 if mode=='endpoint-rotation':
  m=rotation();gap0,gap1,gapmid=m.gap(0),m.gap(1),m.gap(.5);endpoint_accept=abs(gap0)<1e-12 and abs(gap1)<1e-12;result=solve(m)
  failed=endpoint_accept and gapmid<-TOLERANCE and not result['accepted'];print(json.dumps({'negative':'endpoint_only_closed_support','start_gap_m':gap0,'end_gap_m':gap1,'midpoint_gap_m':gapmid,'endpoint_classifier_would_admit':endpoint_accept,'exact_status':result['status'],'rejected_classifier':failed}));return 1 if failed else 2
 if mode=='remaining-frontier':
  m=rotation();result=solve(m);print(json.dumps({'frontier':'rotating_plane_midpoint_penetration','result':result,'midpoint_gap_m':m.gap(.5)}));return 1 if not result['accepted'] else 2
 if mode=='mutable-binding':
  # Enter the actual retained pre-repair implementation, not an approximation
  # of its classifier. Its source/binary/receipt are bound in the history.
  path=HERE/'history/563f6438-before-binding-repair/closed_support.py'
  spec=importlib.util.spec_from_file_location('pre_binding_repair',path);old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old)
  m=retained();historical=old.Motion(m.start,m.end,m.radius)
  for candidate in (m,historical):
   for poses in (candidate.start,candidate.end):
    for owner in poses:owner[2]+=4096
  prior=old.solve(historical);fixed=solve(m);gap=m.gap(.5)
  failed=prior['accepted'] and prior.get('global_nonpenetration_exact') and gap<-TOLERANCE and fixed['status']=='invalid' and not fixed['accepted']
  print(json.dumps({'negative':'mutable_owner_representation_binding','retained_core_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'retained_status':prior['status'],'retained_accepted':prior['accepted'],'actual_narrowed_gap_m':gap,'repaired_status':fixed['status'],'rejected_classifier':failed}));return 1 if failed else 2
 raise AssertionError('unknown mode')

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--negative',choices=('nan-plane','rounded-penetration','endpoint-rotation','remaining-frontier','mutable-binding'));args=parser.parse_args()
 try:exit_code=negative(args.negative) if args.negative else run()
 except Exception as error:print('FAIL',repr(error),file=sys.stderr);exit_code=2
 sys.exit(exit_code)
