#!/usr/bin/env python3
from fractions import Fraction as Q
from decimal import Decimal,localcontext
from functools import cmp_to_key
from pathlib import Path
import json,math,copy
import closed_support as c
import qualify as fixtures
HERE=Path(__file__).resolve().parent

def trim(a):
 a=list(a)
 while len(a)>1 and a[-1]==0:a.pop()
 return a

def add(a,b):return trim([(a[i] if i<len(a) else Q(0))+(b[i] if i<len(b) else Q(0)) for i in range(max(len(a),len(b)))])
def scale(a,b):return trim([x*b for x in a])
def sub(a,b):return add(a,scale(b,-1))
def mul(a,b):
 r=[Q(0)]*(len(a)+len(b)-1)
 for i,x in enumerate(a):
  for j,y in enumerate(b):r[i+j]+=x*y
 return trim(r)
def rem(a,p):
 r=trim(a);p=trim(p)
 while len(r)>=len(p):
  k=len(r)-len(p);factor=r[-1]/p[-1]
  for i,x in enumerate(p):r[k+i]-=factor*x
  r=trim(r)
 return r

def val(a,t):return sum(x*t**i for i,x in enumerate(a))
def affine_range(a,l,h):
 assert len(a)<=2
 low=val(a,l);high=val(a,h)
 return min(low,high),max(low,high)
def owner(m,i,k):return [c.exact(m.start[i][k]),c.exact(m.end[i][k])-c.exact(m.start[i][k])]

checks=[];algebraic=0;rational=0;planes=0
for record in json.loads((HERE/'controls.json').read_text())['cases']:
 r=record['result']
 if r['accepted']:
  m=c.Motion(record['start'],record['end'],record['radius']);n=tuple(Q(x) for x in r['normal_exact']);nn=sum(x*x for x in n);assert nn>0
  poses=m.qpoints();d=sum((poses[0][0][k]-poses[0][2][k])*n[k] for k in range(3));assert d>0 and d*d-c.exact(m.radius)**2*nn>=0
  for pose in poses:
   plane=sum(pose[2][k]*n[k] for k in range(3))
   assert all(sum(pose[i][k]*n[k] for k in range(3))==plane for i in range(2,5))
   assert all(sum(pose[i][k]*n[k] for k in range(3))-plane==d for i in range(2))
  planes+=1
 if r['status'] not in ('grazing','support'):continue
 m=c.Motion(record['start'],record['end'],record['radius']);material=r['material'];radius=c.exact(m.radius)
 if 'yarn_u_exact' in material:
  u=Q(material['yarn_u_exact']);weights=[Q(x) for x in material['triangle_weights_exact']]
  assert 0<=u<=1 and min(weights)>=0 and sum(weights)==1
  t=Q(r['root_exact']) if r['status']=='grazing' else Q(0)
  d=[(1-u)*val(owner(m,0,k),t)+u*val(owner(m,1,k),t)-sum(weights[i]*val(owner(m,i+2,k),t) for i in range(3)) for k in range(3)]
  assert sum(x*x for x in d)==radius*radius
  rational+=1;checks.append({'case':record['name'],'exact_rational_owner_distance':True});continue
 algebraic+=1
 p=[Q(x) for x in r['root_polynomial_exact']];l,h=[Q(x) for x in r['root_isolation_exact']]
 N=[Q(x) for x in material['parameter_numerator']];D=[Q(x) for x in material['parameter_denominator']]
 assert l<h and val(p,l)*val(p,h)<0
 derivative=lambda t:p[1]+2*p[2]*t
 assert derivative(l)*derivative(h)>0
 discriminant=p[1]*p[1]-4*p[0]*p[2]
 assert math.isqrt(discriminant.numerator)**2!=discriminant.numerator or math.isqrt(discriminant.denominator)**2!=discriminant.denominator
 for predicate,strict in ((D,True),(N,False),(sub(D,N),False)):
  reduced=rem(predicate,p);lo,hi=affine_range(reduced,l,h)
  assert lo>0 if strict else lo>=0,(record['name'],'membership bounds',lo,hi)
 delta=[];kind=material['feature'];index=material['index']
 for k in range(3):
  if kind=='vertex':d=add(mul(sub(owner(m,0,k),owner(m,index+2,k)),D),mul(sub(owner(m,1,k),owner(m,0,k)),N))
  else:
   endpoint,edge=divmod(index,3)
   d=sub(mul(sub(owner(m,endpoint,k),owner(m,edge+2,k)),D),mul(sub(owner(m,(edge+1)%3+2,k),owner(m,edge+2,k)),N))
  delta.append(d)
 squared=scale(mul(D,D),-radius*radius)
 for d in delta:squared=add(squared,mul(d,d))
 remainder=rem(squared,p);assert remainder==[Q(0)],(record['name'],'not exact contact',remainder)
 checks.append({'case':record['name'],'exact_algebraic_owner_distance_remainder':[str(x) for x in remainder],'exact_membership_interval_bounds':True})

# Direct bounded exact-root ordering / predicate controls, independent Decimal oracle.
root_checks=0;maximum_used=0
families=[(Q(-1,8),Q(0),Q(1)),(Q(-1,2),Q(0),Q(1)),(Q(3,16),Q(-1),Q(1)),(Q(1,4)-Q(1,3)*Q(2)**-200,Q(-1),Q(1))]
for p in families:
 b=c.Budget();rs=c.roots(p,b)
 with localcontext() as ctx:
  ctx.prec=160
  a,bb,cc=[Decimal(x.numerator)/Decimal(x.denominator) for x in p]
  disc=(bb*bb-4*cc*a).sqrt();expected=sorted(t for t in ((-bb-disc)/(2*cc),(-bb+disc)/(2*cc)) if 0<=t<=1)
  assert len(rs)==len(expected)
  rs.sort(key=cmp_to_key(c.compare_roots))
  for r,expected_t in zip(rs,expected):
   assert Decimal(r.lo.numerator)/Decimal(r.lo.denominator)<=expected_t<=Decimal(r.hi.numerator)/Decimal(r.hi.denominator)
   for q in (Q(0),Q(1,4),Q(1,2),Q(3,4),Q(1)):
    got=r.compare_q(q);oracle=(expected_t>Decimal(q.numerator)/Decimal(q.denominator))-(expected_t<Decimal(q.numerator)/Decimal(q.denominator));assert got==oracle
    for predicate in ((Q(1,10),Q(-1),Q(1)),(Q(0),Q(1),Q(0)),p):
     if predicate==p:oracle_sign=0
     else:
      value=sum(Decimal(x.numerator)/Decimal(x.denominator)*expected_t**i for i,x in enumerate(predicate));oracle_sign=(value>0)-(value<0)
     assert r.sign_at(predicate)==oracle_sign
     root_checks+=1
  maximum_used=max(maximum_used,b.used)
# Proportional irreducible polynomials denote the same roots without iterative search.
b=c.Budget();a=c.roots((Q(-1,8),Q(0),Q(1)),b)[0];z=c.roots((Q(1,4),Q(0),Q(-2)),b)[0];assert c.compare_roots(a,z)==0;root_checks+=1
# Closer branches exceed cap and remain a bounded failure, not an ordering guess.
b=c.Budget();exhausted=False
try:
 rs=c.roots((Q(1,4)-Q(1,3)*Q(2)**-400,Q(-1),Q(1)),b);c.compare_roots(rs[0],rs[1])
except c.Exhausted:exhausted=True
assert exhausted and b.used==128

extra=[]
for direction in ('incoming','outgoing'):
 m=fixtures.retained()
 for i in range(2):m.start[i][1]=0;m.end[i][1]=0;m.end[i][2]=c.f32(0 if direction=='incoming' else .01)
 r=c.solve(m);assert not r['accepted'] and r['status']=='unsupported_closed_plane';extra.append({'case':'initial_'+direction,'result':r})
for delta in (-1,1):
 m=fixtures.height(delta);r=c.solve(m);assert r['status']==('clear' if delta>0 else 'unsupported_closed_plane');extra.append({'case':'one_ULP_'+str(delta),'result':r})
# Exactly at t=1 projected contact and t=0 support remain inclusive.
m=fixtures.retained();m.end=[[x,c.f32(.3),z] if i<2 else p for i,p in enumerate(m.end) for x,_,z in [p]]
r=c.solve(m);assert r['accepted'] and r['status']=='grazing' and Q(r['root_exact'])==1;extra.append({'case':'endpoint_t1','result':r})
m=fixtures.retained()
for i in range(2):m.start[i][1]=c.f32(.3)
r=c.solve(m);assert r['accepted'] and r['status']=='support';extra.append({'case':'endpoint_t0','result':r})
output={'exact_five_owner_closed_plane_identity_checks':planes,'independent_rational_owner_contact_checks':rational,'independent_algebraic_owner_contact_checks':algebraic,'exact_root_predicate_checks':root_checks,'maximum_close_root_order_nodes':maximum_used,'hard_ordering_cap_control_exhausted':exhausted,'cap_used':b.used,'extra_status_controls':extra,'contact_checks':checks,'failures':0,'no_gpu':True}
(HERE/'independent-controls.json').write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps({k:v for k,v in output.items() if k not in ('contact_checks','extra_status_controls')},sort_keys=True))
