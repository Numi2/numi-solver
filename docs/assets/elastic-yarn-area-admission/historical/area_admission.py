"""Exact-polynomial CPU prototype for represented material-face area admission.
No production integration. GATE matches the frozen strict 1e-8 numerical gate.
"""
from dataclasses import dataclass
from fractions import Fraction as Q
import math
import sys
GATE=1e-8;CAP=128;TIME_WIDTH=Q(1,2**44);SPEED_LIMIT=30.
class Invalid(Exception):pass
class Exhausted(Exception):pass
@dataclass
class Budget:
 used:int=0
 def take(self):
  if type(self.used) is not int or self.used<0:raise Invalid('nonnegative integral proof-node count required')
  if self.used>=CAP:raise Exhausted('128 total temporal proof-node cap')
  self.used+=1

def q(x):return Q.from_float(float(x))
def vector(p):
 if len(p)!=3:raise Invalid('three coordinates required')
 out=tuple(float(x) for x in p)
 if not all(math.isfinite(x) for x in out):raise Invalid('finite represented owners required')
 return out

def plus(a,b):return tuple(x+y for x,y in zip(a,b))
def minus(a,b):return tuple(x-y for x,y in zip(a,b))
def scale(a,s):return tuple(x*s for x in a)
def dot(a,b):return sum((x*y for x,y in zip(a,b)),Q(0))
def cross(a,b):return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])
def trim(p):
 p=list(p)
 while len(p)>1 and p[-1]==0:p.pop()
 return tuple(p)
def evaluate(p,t):
 value=Q(0)
 for a in reversed(p):value=value*t+a
 return value
def derivative(p):return trim(tuple(i*p[i] for i in range(1,len(p))) or (Q(0),))
def mul_poly(a,b):
 p=[Q(0)]*(len(a)+len(b)-1)
 for i,x in enumerate(a):
  for j,y in enumerate(b):p[i+j]+=x*y
 return trim(p)
def divrem(a,b):
 a=list(trim(a));b=trim(b)
 if b==(0,):raise Invalid('zero polynomial divisor')
 result=[Q(0)]*max(1,len(a)-len(b)+1)
 while len(a)>=len(b) and a!=( [Q(0)] ):
  d=len(a)-len(b);coefficient=a[-1]/b[-1];result[d]=coefficient
  for k,x in enumerate(b):a[k+d]-=coefficient*x
  a=list(trim(a))
 return trim(result),trim(a)
def monic(p):return tuple(x/p[-1] for x in p)
def gcd(a,b):
 while b!=(0,):a,b=b,divrem(a,b)[1]
 return monic(a)
def sturm(p):
 p=trim(p)
 if len(p)<=1:return []
 g=gcd(p,derivative(p));squarefree=divrem(p,g)[0];squarefree=monic(squarefree)
 sequence=[squarefree,derivative(squarefree)]
 while sequence[-1]!=(0,):
  remainder=divrem(sequence[-2],sequence[-1])[1]
  if remainder==(0,):break
  # Only positive rescaling preserves an individual Sturm sign.
  remainder=tuple(-x/abs(remainder[-1]) for x in remainder);sequence.append(remainder)
 return sequence

def variation(sequence,t):
 signs=[]
 for p in sequence:
  v=evaluate(p,t)
  if v:signs.append(1 if v>0 else -1)
 return sum(x!=y for x,y in zip(signs,signs[1:]))
def roots_between(sequence,l,h):return variation(sequence,l)-variation(sequence,h)
def float_down(t):
 f=float(t)
 return math.nextafter(f,-math.inf) if q(f)>t else f
def float_up(t):
 f=float(t)
 return math.nextafter(f,math.inf) if q(f)<t else f

def represented_area(x):
 a=minus(x[1],x[0]);b=minus(x[2],x[0]);c=cross(a,b)
 return .5*math.hypot(*c)
@dataclass(frozen=True)
class Motion:
 start:tuple
 end:tuple
 duration:float=1.
 def __post_init__(self):
  try:
   if len(self.start)!=3 or len(self.end)!=3:raise Invalid('three owners required')
   object.__setattr__(self,'start',tuple(vector(p) for p in self.start));object.__setattr__(self,'end',tuple(vector(p) for p in self.end))
   duration=float(self.duration)
   if not math.isfinite(duration) or not duration>0:raise Invalid('positive finite duration required')
   object.__setattr__(self,'duration',duration)
  except (TypeError,ValueError,OverflowError) as e:raise Invalid('malformed motion') from e
 def exact(self):return tuple(tuple(q(x) for x in p) for p in self.start),tuple(tuple(q(x) for x in p) for p in self.end)
 def point(self,t):
  a,b=self.exact();return tuple(tuple(float(x+t*(y-x)) for x,y in zip(p,r)) for p,r in zip(a,b))

def area_polynomial(m):
 a,b=m.exact();e1=minus(a[1],a[0]);e2=minus(a[2],a[0]);d1=minus(minus(b[1],b[0]),e1);d2=minus(minus(b[2],b[0]),e2)
 c0=cross(e1,e2);c1=plus(cross(e1,d2),cross(d1,e2));c2=cross(d1,d2)
 p=[Q(0)]*5
 for k in range(3):
  term=mul_poly((c0[k],c1[k],c2[k]),(c0[k],c1[k],c2[k]))
  for i,z in enumerate(term):p[i]+=z
 p[0]-=4*q(GATE)**2
 return trim(p)

def cast(m,budget=None):
 budget=Budget() if budget is None else budget
 result={'status':'unresolved','accepted':False,'gate_m2':GATE}
 try:
  budget.take();p=area_polynomial(m);result['polynomial_exact']=[str(x) for x in p]
  if evaluate(p,Q(0))<=0 or not math.isfinite(represented_area(m.start)) or represented_area(m.start)<=GATE:
   result['status']='initial_area_violation';return result
  sequence=sturm(p);number=roots_between(sequence,Q(0),Q(1)) if sequence else 0
  if not number:
   if not math.isfinite(represented_area(m.end)) or represented_area(m.end)<=GATE:
    result['reason']='represented terminal metric lacks strict margin';return result
   result.update(status='clear',accepted=True,strict_polynomial_throughout=True,lower_time_exact='1',upper_time_exact='1',root_count=0);return result
  l,h=Q(0),Q(1)
  while h-l>TIME_WIDTH:
   budget.take();mid=(l+h)/2
   if roots_between(sequence,l,mid)>0:h=mid
   else:l=mid
  result.update(status='first_boundary',lower_time=float_down(l),upper_time=float_up(h),lower_time_exact=str(l),upper_time_exact=str(h),
                first_root_in_bracket_exact=True,no_earlier_root_exact=True,root_count=number,bracket_roots=roots_between(sequence,l,h),requested_time_width_exact=str(TIME_WIDTH))
  return result
 except (Exhausted,Invalid) as e:result['reason']=str(e);return result
 finally:result['total_proof_nodes']=budget.used

def kinetic_exact(v,m):return sum((mass*dot(velocity,velocity)/2 for mass,velocity in zip(m,v)),Q(0))
def within_scene_speed(v):
 # The owning source requires maximumSpeed < 30.0. Both exact represented
 # norm and the source-like floating metric must pass; equality is rejected.
 return all(dot(tuple(q(z) for z in speed),tuple(q(z) for z in speed))<q(SPEED_LIMIT)**2 and math.hypot(*speed)<SPEED_LIMIT for speed in v)
def passive_velocity(x,velocities,masses,pins=()):
 result={'status':'invalid','accepted':False}
 try:
  x=tuple(vector(p) for p in x);v=tuple(vector(p) for p in velocities);m=tuple(float(z) for z in masses);pins=tuple(pins)
  if len(x)!=3 or len(v)!=3 or len(m)!=3 or any(not math.isfinite(z) or z<=0 for z in m):raise Invalid('finite positive masses and three owners required')
  if any(type(i) is not int or i<0 or i>=3 for i in pins) or len(set(pins))!=len(pins):raise Invalid('valid distinct pin indices required')
  if any(any(component!=0 for component in v[i]) for i in pins):
   result.update(status='unsupported_moving_pin',reason='signed actuator work is not implemented');return result
  X=tuple(tuple(q(z) for z in p) for p in x);V=tuple(tuple(q(z) for z in p) for p in v);M=tuple(q(z) for z in m)
  c=cross(minus(X[1],X[0]),minus(X[2],X[0]));B=dot(c,c)/4
  if B<q(GATE)**2 or represented_area(x)<GATE:raise Invalid('response face below unchanged gate')
  if not dot(c,c):raise Invalid('zero area normal')
  # h_i = |cross|*g_i = ∇(A²). At positive area this common
  # positive scaling gives precisely the same passive area-rate impulse.
  H=(scale(cross(c,minus(X[2],X[1])),Q(1,2)),scale(cross(c,minus(X[0],X[2])),Q(1,2)),scale(cross(c,minus(X[1],X[0])),Q(1,2)))
  assert plus(plus(H[0],H[1]),H[2])==(0,0,0)
  assert plus(plus(cross(X[0],H[0]),cross(X[1],H[1])),cross(X[2],H[2]))==(0,0,0)
  W=sum((dot(h,h)/mass for i,(h,mass) in enumerate(zip(H,M)) if i not in pins),Q(0));r=sum((dot(h,speed) for h,speed in zip(H,V)),Q(0))
  if W<=0:result.update(status='unresolved',reason='zero free-owner response metric');return result
  p=max(Q(0),-r/W)
  ideal=tuple(plus(speed,scale(h,p/mass)) if i not in pins else speed for i,(speed,h,mass) in enumerate(zip(V,H,M)))
  rounded=tuple(tuple(float(z) for z in speed) for speed in ideal)
  if not all(math.isfinite(z) for speed in rounded for z in speed):raise Invalid('unrepresented output velocity')
  R=tuple(tuple(q(z) for z in speed) for speed in rounded)
  delta=kinetic_exact(R,M)-kinetic_exact(V,M);ideal_delta=kinetic_exact(ideal,M)-kinetic_exact(V,M)
  expected=-min(Q(0),r)**2/(2*W)
  assert ideal_delta==expected
  result.update(represented_kinetic_change_exact_J=str(delta),represented_kinetic_change_J=float(delta),ideal_kinetic_change_exact_J=str(ideal_delta),
                impulse_scale_exact=str(p),area_squared_rate_before_exact=str(r),area_squared_rate_after_represented_exact=str(sum((dot(h,speed) for h,speed in zip(H,R)),Q(0))))
  if delta>0:
   result.update(status='represented_passivity_failure',reason='actual rounded velocity increases kinetic energy; no tolerance admission');return result
  anchor=tuple(sum((p*H[i][k] for i in pins),Q(0)) for k in range(3));anchor_torque=tuple(sum((cross(X[i],scale(H[i],p))[k] for i in pins),Q(0)) for k in range(3))
  actual_P=tuple(sum((M[i]*(R[i][k]-V[i][k]) for i in range(3)),Q(0)) for k in range(3))
  actual_L=tuple(sum((cross(X[i],scale(minus(R[i],V[i]),M[i]))[k] for i in range(3)),Q(0)) for k in range(3))
  residual_P=tuple(actual_P[k]+anchor[k] for k in range(3));residual_L=tuple(actual_L[k]+anchor_torque[k] for k in range(3))
  # Explicit absolute admission envelope for this bounded source domain.
  # It scales with 30 m/s, never with an unbounded common velocity. The ideal
  # impulse is exactly reciprocal; these checks admit only the small error of
  # representing its final velocities. The angular scale is a world-origin
  # radius bound, so it is stated separately from the linear units.
  tolerance_P=32*q(sys.float_info.epsilon)*sum(M,Q(0))*q(SPEED_LIMIT)
  radius=max(Q(1),max(sum((abs(z) for z in owner),Q(0)) for owner in X))
  tolerance_L=tolerance_P*radius
  result.update(velocities=rounded,
    constraint_impulse_on_anchors_exact=[str(z) for z in anchor],external_support_impulse_on_free_body_exact=[str(-z) for z in anchor],
    constraint_angular_impulse_on_anchors_exact=[str(z) for z in anchor_torque],external_support_angular_impulse_exact=[str(-z) for z in anchor_torque],
    represented_linear_balance_residual_exact=[str(z) for z in residual_P],represented_angular_balance_residual_exact=[str(z) for z in residual_L],
    represented_linear_balance_tolerance_exact=str(tolerance_P),represented_angular_balance_tolerance_exact=str(tolerance_L),
    strict_scene_speed_limit_m_s=SPEED_LIMIT)
  if any(abs(z)>tolerance_P for z in residual_P) or any(abs(z)>tolerance_L for z in residual_L):
   result.update(status='represented_reciprocity_failure',reason='actual rounded momentum or angular balance exceeds the stated source-domain envelope');return result
  if not within_scene_speed(v) or not within_scene_speed(rounded):
   result.update(status='scene_speed_failure',reason='input or represented response speed lacks strict 30 m/s source margin');return result
  result.update(status='passive',accepted=True)
  return result
 except (Invalid,TypeError,ValueError,OverflowError) as e:result.update(status='invalid',reason=str(e));return result

def preventive(m,masses,pins=()):
 budget=Budget();result={'status':'invalid','accepted':False}
 try:
  masses=tuple(float(z) for z in masses);pins=tuple(pins)
  if len(masses)!=3 or any(not math.isfinite(z) or z<=0 for z in masses):raise Invalid('finite positive masses required')
  if any(type(i) is not int or i<0 or i>=3 for i in pins) or len(set(pins))!=len(pins):raise Invalid('valid distinct pin indices required')
  a,b=m.exact();duration=q(m.duration)
  exact_velocities=tuple(tuple((y-x)/duration for x,y in zip(p,r)) for p,r in zip(a,b))
  velocities=tuple(tuple(float(z) for z in speed) for speed in exact_velocities)
  if not all(math.isfinite(z) for speed in velocities for z in speed):raise Invalid('unrepresented proposal velocity')
  if any(a[i]!=b[i] for i in pins):
   result.update(status='unsupported_moving_pin',reason='signed actuator work is not implemented',total_proof_nodes=0);return result
  if any(dot(speed,speed)>=q(SPEED_LIMIT)**2 for speed in exact_velocities) or not within_scene_speed(velocities):
   result.update(status='scene_speed_failure',reason='proposal speed lacks strict 30 m/s source margin',total_proof_nodes=0);return result
 except (Invalid,TypeError,ValueError,OverflowError) as e:
  result.update(reason=str(e),total_proof_nodes=0);return result
 first=cast(m,budget);result={'status':'unresolved','accepted':False,'original_cast':first}
 if first['status']=='clear':result.update(status='clear',accepted=True,endpoint=m.end,total_proof_nodes=budget.used);return result
 if first['status']!='first_boundary':result['reason']=first['status'];result['total_proof_nodes']=budget.used;return result
 t=float_down(Q(first['lower_time_exact']))
 if t==1:t=math.nextafter(t,0.)
 try:
  for attempt in range(16):
   budget.take();point=m.point(q(t));incoming=cast(Motion(m.start,point,m.duration),budget)
   if incoming['status']=='clear' and represented_area(point)>GATE:break
   if t==0:raise Exhausted('no represented strict boundary prefix')
   t=math.nextafter(t,0.)
  else:raise Exhausted('16 represented boundary retries exhausted')
  response=passive_velocity(point,velocities,masses,pins);result.update(boundary_time=t,boundary_positions=point,boundary_represented_area_m2=represented_area(point),incoming_prefix=incoming,response=response)
  if not response['accepted']:result['reason']=response['status'];return result
  remaining=(1-q(t))*duration
  endpoint=tuple(tuple(float(q(x)+q(v)*remaining) for x,v in zip(p,speed)) for p,speed in zip(point,response['velocities']))
  post=cast(Motion(point,endpoint,float(remaining)),budget);result['post_response_cast']=post;result['endpoint']=endpoint
  if post['status']=='clear':result.update(status='admitted_response',accepted=True)
  else:result['reason']='post-response linear owner path not certified; owned subdivision/rollback required'
  return result
 except (Exhausted,Invalid,ValueError,OverflowError) as e:result['reason']=str(e);return result
 finally:result['total_proof_nodes']=budget.used
