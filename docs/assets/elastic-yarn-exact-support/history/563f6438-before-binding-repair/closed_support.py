"""Isolated exact closed-plane / algebraic planar CCD certificate.
No production writes, GPU, response impulses or trajectories.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from fractions import Fraction as Q
from functools import cmp_to_key
from math import copysign, isfinite, isqrt, nextafter, sqrt, inf
import ctypes, json, struct
from pathlib import Path

TOLERANCE = 2e-6
OUTER_BUDGET = 128
HERE = Path(__file__).resolve().parent
BRIDGE = ctypes.CDLL(str(HERE / 'oracle_bridge.dylib'))
BRIDGE.independent_gap.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_double]
BRIDGE.independent_gap.restype = ctypes.c_double
BRIDGE.interval_motion_upper.argtypes = [ctypes.POINTER(ctypes.c_float)]
BRIDGE.interval_motion_upper.restype = ctypes.c_double
BRIDGE.admit_face.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_uint, ctypes.POINTER(ctypes.c_uint)]
BRIDGE.admit_face.restype = ctypes.c_int
for name in ('legacy_plane_lower','safe_plane_lower'):
    helper = getattr(BRIDGE,name)
    helper.argtypes = [ctypes.POINTER(ctypes.c_float),ctypes.c_double,ctypes.c_double,ctypes.POINTER(ctypes.c_double)]
    helper.restype = ctypes.c_double

class Exhausted(Exception): pass
class Invalid(Exception): pass

def f32(x):
    try: v = struct.unpack('f', struct.pack('f', x))[0]
    except (OverflowError, struct.error): raise Invalid('unrepresented FP32 owner')
    if not isfinite(v): raise Invalid('nonfinite owner')
    return v

def exact(x):
    if not isfinite(x): raise Invalid('nonfinite exact input')
    return Q.from_float(float(x))

def sign(x): return (x > 0) - (x < 0)
def add(a,b): return tuple(x+y for x,y in zip(a,b))
def sub(a,b): return tuple(x-y for x,y in zip(a,b))
def scale(a,s): return tuple(x*s for x in a)
def dot(a,b): return sum((x*y for x,y in zip(a,b)), Q(0))
def cross(a,b): return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])
def cross2(a,b): return a[0]*b[1]-a[1]*b[0]
def orient(a,b,c): return cross2(sub(b,a),sub(c,a))
def poly(p,t): return p[0]+t*(p[1]+t*p[2])
def polysub(p,q): return tuple(x-y for x,y in zip(p,q))
def canonical(p):
    divisor = p[2] if p[2] else p[1] if p[1] else p[0]
    return tuple(x/divisor for x in p) if divisor else p

def determinant(a,b):
    return (cross2(a[0],b[0]), cross2(a[0],b[1])+cross2(a[1],b[0]), cross2(a[1],b[1]))
def product(a,b):
    return (dot(a[0],b[0]), dot(a[0],b[1])+dot(a[1],b[0]), dot(a[1],b[1]))
def moving_sub(a,b): return (sub(a[0],b[0]),sub(a[1],b[1]))
def moving_orient(a,b,c): return determinant(moving_sub(b,a),moving_sub(c,a))
def moving_point(p,t): return add(p[0],scale(p[1],t))

def qfloat_down(q):
    x=float(q)
    return nextafter(x,-inf) if exact(x)>q else x

def qfloat_up(q):
    x=float(q)
    return nextafter(x,inf) if exact(x)<q else x

@dataclass
class Budget:
    used: int = 0
    maximum: int = OUTER_BUDGET
    def take(self):
        if self.used >= self.maximum: raise Exhausted('128 event/interval-node cap')
        self.used += 1

@dataclass
class Root:
    p: tuple[Q,Q,Q]
    lo: Q
    hi: Q
    budget: Budget = field(repr=False)
    @property
    def rational(self): return self.lo if self.lo==self.hi else None
    def refine(self):
        self.budget.take()
        middle=(self.lo+self.hi)/2
        left,center=poly(self.p,self.lo),poly(self.p,middle)
        if center==0: self.lo=self.hi=middle
        elif sign(left)==sign(center): self.lo=middle
        else: self.hi=middle
    def compare_q(self,q):
        if self.rational is not None: return sign(self.lo-q)
        if q < self.lo: return 1
        if q > self.hi: return -1
        value=poly(self.p,q)
        if value==0: return 0
        derivative=sign(2*self.p[2]*((self.lo+self.hi)/2)+self.p[1])
        # Exact monotonic root isolation: increasing p(q)>0 means q is after root.
        return -sign(value)*derivative
    def sign_at(self,q):
        if self.rational is not None: return sign(poly(q,self.lo))
        # Any degree<=2 predicate reduced modulo the irreducible root polynomial
        # is affine, whose sign is determined by one exact root/rational comparison.
        factor=q[2]/self.p[2]
        r0=q[0]-factor*self.p[0];r1=q[1]-factor*self.p[1]
        if r1==0: return sign(r0)
        return sign(r1)*self.compare_q(-r0/r1)
    def key_branch(self):
        c=canonical(self.p)
        return c,sign(2*c[2]*((self.lo+self.hi)/2)+c[1])

def compare_roots(a,b):
    if a.rational is not None: return -b.compare_q(a.lo)
    if b.rational is not None: return a.compare_q(b.lo)
    if a.key_branch()==b.key_branch(): return 0
    while not (a.hi < b.lo or b.hi < a.lo):
        (a if a.hi-a.lo >= b.hi-b.lo else b).refine()
    return -1 if a.hi < b.lo else 1

def square_root_q(q):
    a,b=isqrt(q.numerator),isqrt(q.denominator)
    return Q(a,b) if a*a==q.numerator and b*b==q.denominator else None

def roots(p,budget):
    if p[2]==0:
        if p[1]==0: return []
        t=-p[0]/p[1]
        return [Root(p,t,t,budget)] if 0<=t<=1 else []
    d=p[1]*p[1]-4*p[2]*p[0]
    if d<0: return []
    rational_sqrt=square_root_q(d)
    if rational_sqrt is not None:
        ts=sorted(set(((-p[1]-rational_sqrt)/(2*p[2]),(-p[1]+rational_sqrt)/(2*p[2]))))
        return [Root(p,t,t,budget) for t in ts if 0<=t<=1]
    vertex=-p[1]/(2*p[2]);cuts=sorted(set((Q(0),Q(1),min(Q(1),max(Q(0),vertex)))))
    result=[]
    divisor=max(abs(x) for x in p)
    cf,bf,af=(float(x/divisor) for x in p)
    df=float(d/(divisor*divisor))
    guesses=[]
    if df>0 and isfinite(df):
        q=-.5*(bf+copysign(sqrt(df),bf))
        if af and q: guesses=sorted((q/af,cf/q))
    for left,right in zip(cuts,cuts[1:]):
        fl,fr=poly(p,left),poly(p,right)
        if fl==0: result.append(Root(p,left,left,budget));continue
        if fr==0: result.append(Root(p,right,right,budget));continue
        if sign(fl)==sign(fr): continue
        root=Root(p,left,right,budget)
        # Guesses can only shrink a root interval after exact polynomial signs
        # and the monotone branch prove containment. Never accept sqrt rounding.
        for guess in guesses:
            if not isfinite(guess) or not float(left)<=guess<=float(right): continue
            low=nextafter(guess,-inf);high=nextafter(guess,inf)
            for _ in range(16):
                budget.take()
                l=max(left,exact(low));h=min(right,exact(high))
                if l<=h and poly(p,l)==0: root.lo=root.hi=l;break
                if l<=h and poly(p,h)==0: root.lo=root.hi=h;break
                if l<h and sign(poly(p,l))!=sign(poly(p,h)):
                    root.lo,root.hi=l,h;break
                low=nextafter(low,-inf);high=nextafter(high,inf)
            if root.lo!=left or root.hi!=right: break
        result.append(root)
    return result

@dataclass
class Feature:
    kind: str
    index: int
    p: tuple[Q,Q,Q]
    numerator: tuple[Q,Q,Q]
    denominator: tuple[Q,Q,Q]
    other: tuple[Q,Q,Q]
    def feasible(self,root):
        return root.sign_at(self.denominator)>0 and root.sign_at(self.numerator)>=0 and root.sign_at(self.other)>=0
    def material(self,root):
        if root.rational is not None:
            value=poly(self.numerator,root.lo)/poly(self.denominator,root.lo)
            if not 0<=value<=1: raise AssertionError('non-feasible exact material coordinate')
            if self.kind=='vertex': return {'yarn_u_exact':str(value),'triangle_weights_exact':[str(int(i==self.index)) for i in range(3)]}
            endpoint,edge=divmod(self.index,3)
            weights=[Q(0)]*3;weights[edge]=1-value;weights[(edge+1)%3]=value
            return {'yarn_u_exact':str(endpoint),'triangle_weights_exact':[str(x) for x in weights]}
        # Algebraic coordinates remain descriptor-valued. Predicates prove
        # membership at the exact root; no rounded coordinates pretend contact.
        return {'feature':self.kind,'index':self.index,'parameter_numerator':[str(x) for x in self.numerator],
                'parameter_denominator':[str(x) for x in self.denominator],
                'feasible_at_exact_root':True,'represented_material_point_claimed':False}

@dataclass
class Motion:
    start: list
    end: list
    radius: float=.004
    def __post_init__(self):
        if len(self.start)!=5 or len(self.end)!=5 or any(len(p)!=3 for p in self.start+self.end): raise Invalid('owner count')
        self.start=[[f32(x) for x in p] for p in self.start];self.end=[[f32(x) for x in p] for p in self.end];self.radius=f32(self.radius)
        if self.radius<=0: raise Invalid('nonpositive radius')
    def data(self): return (ctypes.c_float*31)(*[x for p in self.start+self.end for x in p],self.radius)
    def qpoints(self): return [[tuple(exact(x) for x in p) for p in poses] for poses in (self.start,self.end)]
    def gap(self,t): return BRIDGE.independent_gap(self.data(),float(t))

def closed_plane(m):
    start,end=m.qpoints();n=cross(sub(start[3],start[2]),sub(start[4],start[2]));nn=dot(n,n)
    if nn==0: return None
    d=dot(sub(start[0],start[2]),n)
    if d<0: n=scale(n,-1);d=-d
    if d<=0: return None
    for points in (start,end):
        plane=dot(points[2],n)
        if any(dot(points[i],n)!=plane for i in range(2,5)): return None
        if any(dot(points[i],n)-plane!=d for i in range(2)): return None
    comparison=sign(d*d-exact(m.radius)**2*nn)
    if comparison<0: return None
    # Orthogonal projection, not coordinate dropping alone: oblique support
    # requires subtracting d*n/||n||^2 exactly before the 2D planar predicates.
    shift=scale(n,d/nn);axis=max(range(3),key=lambda k:abs(n[k]));keep=[k for k in range(3) if k!=axis]
    moving=[]
    for i in range(5):
        a=sub(start[i],shift) if i<2 else start[i];b=sub(end[i],shift) if i<2 else end[i]
        aa=tuple(a[k] for k in keep);bb=tuple(b[k] for k in keep)
        moving.append((aa,sub(bb,aa)))
    return n,moving,comparison

def initial_pair(moving):
    p=[x[0] for x in moving];area=orient(p[2],p[3],p[4])
    for i in range(2):
        v=orient(p[2],p[i],p[4])/area;w=orient(p[2],p[3],p[i])/area
        if v>=0 and w>=0 and v+w<=1: return {'yarn_u_exact':str(i),'triangle_weights_exact':[str(1-v-w),str(v),str(w)]}
    axis=sub(p[1],p[0])
    for i in range(3):
        a,b=p[2+i],p[2+(i+1)%3];edge=sub(b,a);det=cross2(axis,edge)
        if det:
            u=cross2(sub(a,p[0]),edge)/det;v=cross2(sub(a,p[0]),axis)/det
            if 0<=u<=1 and 0<=v<=1:
                weights=[Q(0)]*3;weights[i]=1-v;weights[(i+1)%3]=v
                return {'yarn_u_exact':str(u),'triangle_weights_exact':[str(x) for x in weights]}
        elif orient(p[0],p[1],a)==0 and dot(axis,axis)>0:
            for j,point in ((i,a),((i+1)%3,b)):
                u=dot(sub(point,p[0]),axis)/dot(axis,axis)
                if 0<=u<=1: return {'yarn_u_exact':str(u),'triangle_weights_exact':[str(int(k==j)) for k in range(3)]}
    return None

def features(moving):
    result=[]
    for i in range(3):
        axis=moving_sub(moving[1],moving[0]);offset=moving_sub(moving[2+i],moving[0]);d=product(axis,axis);n=product(offset,axis)
        result.append(Feature('vertex',i,moving_orient(moving[0],moving[1],moving[2+i]),n,d,polysub(d,n)))
    for endpoint in range(2):
        for edge in range(3):
            a,b=moving[2+edge],moving[2+(edge+1)%3];axis=moving_sub(b,a);offset=moving_sub(moving[endpoint],a);d=product(axis,axis);n=product(offset,axis)
            result.append(Feature('endpoint',endpoint*3+edge,moving_orient(a,b,moving[endpoint]),n,d,polysub(d,n)))
    return result

def solve(m,maximum=OUTER_BUDGET):
    budget=Budget(maximum=maximum)
    result={'status':'unresolved','lower_time':0.,'upper_time':1.,'outer_nodes':0,'accepted':False,'tolerance_m':TOLERANCE}
    try:
        if not isinstance(maximum,int) or maximum<0 or maximum>OUTER_BUDGET:
            result["status"]="invalid";return result
        used=ctypes.c_uint();admission=BRIDGE.admit_face(m.data(),maximum,ctypes.byref(used));budget.used=used.value
        if admission: result['status']='degenerate' if admission==1 else 'unresolved';return result
        plane=closed_plane(m)
        if plane is None: result['status']='unsupported_closed_plane';return result
        n,moving,comparison=plane
        result['normal_exact']=[str(x) for x in n];result['global_nonpenetration_exact']=True
        result['normal_closing_exact']='0'
        if comparison>0: result.update(status='clear',accepted=True,lower_time=1.,upper_time=1.);return result
        start=initial_pair(moving)
        if start is not None: result.update(status='support',accepted=True,lower_time=0.,upper_time=0.,material=start);return result
        candidates=[]
        for feature in features(moving):
            budget.take()
            polys=[feature.p] if any(feature.p) else [feature.numerator,feature.other]
            for p in polys:
                if not any(p): continue
                for root in roots(p,budget):
                    if root.compare_q(Q(0))<0 or root.compare_q(Q(1))>0: raise AssertionError('root outside [0,1]')
                    if feature.feasible(root): candidates.append((root,feature))
        if not candidates: result.update(status='clear',accepted=True,lower_time=1.,upper_time=1.,no_feasible_boundary_root_exact=True);return result
        candidates.sort(key=cmp_to_key(lambda a,b:compare_roots(a[0],b[0])))
        root,feature=candidates[0]
        motion_upper=BRIDGE.interval_motion_upper(m.data())
        if not isfinite(motion_upper): raise Exhausted('nonfinite motion upper bound')
        while True:
            lower,upper=qfloat_down(root.lo),qfloat_up(root.hi)
            spatial=0. if upper==lower else nextafter(nextafter(upper-lower,inf)*motion_upper,inf)
            if spatial<=TOLERANCE: break
            root.refine()
        # The exact root is feasible. Before it no boundary entry exists, and
        # the entire time path has an exact closed-plane no-penetration proof.
        # At the lower bracket point, frozen event coordinates yield an upper
        # gap bounded by the motion norm times root-bracket width.
        result.update(status='grazing',accepted=True,lower_time=lower,upper_time=upper,
          temporal_spatial_width_upper_m=spatial,gap_at_lower_upper_m=spatial,
          root_polynomial_exact=[str(x) for x in root.p],root_isolation_exact=[str(root.lo),str(root.hi)],
          root_exact=str(root.rational) if root.rational is not None else None,
          material=feature.material(root),feature=feature.kind,feature_index=feature.index,
          clear_prefix_exact=True,contact_at_algebraic_root_exact=True,
          represented_penetration_used=False,response_impulse_claimed=False)
        return result
    except Exhausted as e: result['frontier']=str(e);return result
    finally: result['outer_nodes']=budget.used
