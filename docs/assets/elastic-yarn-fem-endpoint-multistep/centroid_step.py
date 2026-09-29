#!/usr/bin/env python3
"""New FP64 support-Verlet/contact split of a represented complete native FEM state.
No native bit-replay or CCD claim. All geometry uses these actual FP64 owners.
"""
import copy, hashlib, itertools, json, math, pathlib, struct, sys, time
from decimal import Decimal as D, localcontext

P=pathlib.Path(__file__).resolve().parent
ROOT=P.parents[1]; PACK=ROOT/'build/elastic-yarn-fem-step-preparation'
EPS=2.0**-52; EPS32=2.0**-23
def f32(x): return struct.unpack('<f',struct.pack('<f',x))[0]
J_GATE=f32(1e-6); REST_GATE=f32(2e-4); GAP_GATE=2e-6
G=(0.,0.,f32(-9.81)); DT=1e-6
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def add(a,b): return [a[k]+b[k] for k in range(3)]
def sub(a,b): return [a[k]-b[k] for k in range(3)]
def mul(a,b): return [x*b for x in a]
def dot(a,b): return math.fsum(x*y for x,y in zip(a,b))
def cross(a,b): return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]
def norm(a): return math.sqrt(dot(a,a))
def sumv(a): return [math.fsum(q[k] for q in a) for k in range(3)]
def det(a): return dot(a[0],cross(a[1],a[2]))
def matmul(a,b): return [[dot(a[r],[b[k][c] for k in range(3)]) for c in range(3)] for r in range(3)]
def transpose(a): return [list(q) for q in zip(*a)]
def finite(xs): return all(math.isfinite(x) for x in xs)
class Reject(Exception):
    def __init__(self, reason, detail=None): self.reason=reason; self.detail=detail; super().__init__(reason)

class Model:
    def __init__(self):
        self.body=json.loads((PACK/'body.json').read_text())
        self.prov=json.loads((ROOT/'docs/assets/elastic-yarn-fem-frame38-provenance.json').read_text())
        receipt=json.loads((PACK/'receipt.json').read_text())
        for name,expected in receipt['sha256'].items():
            assert sha(ROOT/name)==expected,(name,'immutable pack changed')
        pack_evidence=json.loads((PACK/'evidence.json').read_text())
        for name,expected in pack_evidence['outputs_sha256'].items():
            assert sha(pathlib.Path(name))==expected,(name,'immutable pack payload changed')
        self.inputs={str(PACK/'body.json'):sha(PACK/'body.json'), str(PACK/'receipt.json'):sha(PACK/'receipt.json'),
          str(ROOT/'docs/assets/elastic-yarn-fem-frame38-provenance.json'):sha(ROOT/'docs/assets/elastic-yarn-fem-frame38-provenance.json')}
        for path in (PACK/'frozen').rglob('*'):
            if path.is_file(): self.inputs[str(path)]=sha(path)
        self.nodes=self.body['nodes']; self.elements=self.body['elements']; self.n=len(self.nodes)
        assert self.n==2057 and len(self.elements)==10240
        self.mass=[q['lumped_mass_kg_f32'] for q in self.nodes]+[q['mass'] for q in self.prov['states'][:2]]
        assert all(math.isfinite(m) and m>0 for m in self.mass)
        self.radius=self.prov['yarn_radius_m']; assert self.radius==f32(self.radius)>0
        self.row=[self.n,self.n+1]+self.body['selected_boundary_face']
        self.beta=[0.33333331346511841,0.3333333432674408,0.3333333432674408]
        self.weight=[.5,.5]+[-q for q in self.beta]; assert math.fsum(self.weight)==0
        for k,node in enumerate(self.row[2:]):
            q=self.nodes[node]; z=self.prov['states'][k+2]
            assert q['position_f32']==z['position'] and q['velocity_f32']==z['velocity'] and q['lumped_mass_kg_f32']==z['mass']
        self.max_rest_error=0.
        for e,q in enumerate(self.elements):
            b=q['inverse_rest_rows_f32']; v=q['rest_volume_m3_f32']; mu,lam=q['material_f32'][:2]
            residual=abs(6*v*det(b)-1); self.max_rest_error=max(self.max_rest_error,residual)
            if not finite(sum(b,[])+[v,mu,lam]) or v<=0 or det(b)<=0 or residual>REST_GATE or mu<=0 or lam<0 or q['control_u32']!=[1,0,0,0]:
                raise Reject('rest/material/ABI',{'element':e})
        assert self.row==[2057,2058,5,1814,1347]
    def initial(self):
        return {'x':[q['position_f32'][:] for q in self.nodes]+[q['position'][:] for q in self.prov['states'][:2]],
          'v':[q['velocity_f32'][:] for q in self.nodes]+[q['velocity'][:] for q in self.prov['states'][:2]],
          'elapsed':0.,'steps':0,'warm':{'radius':self.radius,'u':.5,'beta':self.beta[:]},'ledger':[], 'cache':{}}
    def evaluate(self,x,retain=True):
        if len(x)!=self.n+2 or not all(finite(q) for q in x): raise Reject('nonfinite/owner-count')
        force=[[0.,0.,0.] for _ in x]; energies=[]; outputs=[]; minj=math.inf; maxj=-math.inf; mini=-1
        for index,q in enumerate(self.elements):
            ns=q['nodes']; edge=[sub(x[ns[k]],x[ns[0]]) for k in (1,2,3)]
            b=q['inverse_rest_rows_f32']; vol=q['rest_volume_m3_f32']; mu,lam=q['material_f32'][:2]
            F=matmul(transpose(edge),b); j=det(F)
            if not math.isfinite(j) or j<=J_GATE: raise Reject('invalid-J',{'element':index,'J':j,'gate':J_GATE})
            fc=transpose(F); co=transpose([cross(fc[1],fc[2]),cross(fc[2],fc[0]),cross(fc[0],fc[1])])
            logj=math.log(j); factor=(lam*logj-mu)/j
            pi=[[mu*F[r][c]+factor*co[r][c] for c in range(3)] for r in range(3)]
            energy=vol*(.5*mu*(math.fsum(z*z for z in sum(F,[]))-3)-mu*logj+.5*lam*logj*logj)
            H=[[ -vol*z for z in row] for row in matmul(pi,transpose(b))]
            h=transpose(H); local=[mul(sumv(h),-1)]+h
            if not finite(sum(pi,[])+sum(local,[])+[energy]): raise Reject('nonfinite-stress',{'element':index})
            # Same element/corner order as the source's sorted incidence gather.
            for corner,node in enumerate(ns):
                for k in range(3): force[node][k]+=local[corner][k]
            energies.append(energy)
            if retain: outputs.append(sum(pi,[])+[j,energy])
            if j<minj: minj=j; mini=index
            maxj=max(maxj,j)
        return {'force':force,'U':math.fsum(energies),'minJ':minj,'maxJ':maxj,'min_element':mini,'outputs':outputs,
          'net_internal_force_N':sumv(force),'net_internal_torque_Nm':sumv([cross(x[i],force[i]) for i in range(self.n)])}
    def geometry(self,x):
        own=[x[i] for i in self.row]; a,b,*t=own
        edge=[sub(t[1],t[0]),sub(t[2],t[0]),sub(t[2],t[1])]; maxe=max(dot(q,q) for q in edge)
        cross2=dot(cross(edge[0],edge[1]),cross(edge[0],edge[1])); gate=128*EPS32*EPS32*maxe*maxe
        if not finite(sum(own,[])) or cross2<=gate or maxe<=0: raise Reject('invalid-contact-area',{'cross_squared':cross2,'gate':gate})
        yp=mul(add(a,b),.5); tp=sumv([mul(t[i],self.beta[i]) for i in range(3)])
        delta=sub(yp,tp); dist=norm(delta)
        threshold=8*EPS32*max(1.,norm(yp),norm(tp))
        if dist<=threshold: raise Reject('undefined-contact-normal',{'distance':dist,'threshold':threshold})
        oracle=closest(a,b,t)
        if not oracle['valid']: raise Reject('oracle-unresolved')
        gap=oracle['distance']-self.radius
        if gap< -GAP_GATE or abs(dist-oracle['distance'])>GAP_GATE: raise Reject('contact-manifold/gap',{'row_distance':dist,'oracle':oracle,'radius':self.radius})
        return {'distance':dist,'global_distance':oracle['distance'],'gap':gap,'row_gap':dist-self.radius,'normal':mul(delta,1/dist),
          'u':.5,'beta':self.beta[:],'oracle':oracle,'triangle_area_m2':.5*math.sqrt(cross2),'radius':self.radius}
    def impulse(self,s,mode='correct'):
        geometry=self.geometry(s['x']); n=geometry['normal']; grad=[mul(n,w) for w in self.weight]
        c=math.fsum(dot(grad[k],s['v'][node]) for k,node in enumerate(self.row))
        inv=math.fsum(dot(grad[k],grad[k])/self.mass[node] for k,node in enumerate(self.row))
        impulse=max(0.,-c/inv) if geometry['row_gap']<=GAP_GATE else 0.
        if mode=='energy-positive': impulse*=3
        old=[s['v'][node][:] for node in self.row]
        for k,node in enumerate(self.row):
            if mode=='disabled-body' and k>=2: continue
            s['v'][node]=add(s['v'][node],mul(grad[k],impulse/self.mass[node]))
        new=[s['v'][node][:] for node in self.row]
        with localcontext() as ctx:
            ctx.prec=80
            dk=sum(D.from_float(self.mass[node])/2*sum((D.from_float(new[k][j])-D.from_float(old[k][j]))*(D.from_float(new[k][j])+D.from_float(old[k][j])) for j in range(3)) for k,node in enumerate(self.row))
        impulses=[mul(sub(new[k],old[k]),self.mass[node]) for k,node in enumerate(self.row)]
        dp=sumv(impulses); dl=sumv([cross(s['x'][node],impulses[k]) for k,node in enumerate(self.row)])
        cb=math.fsum(dot(grad[k],new[k]) for k in range(5))
        scale=math.fsum(norm(q) for q in impulses)
        # A rounded v_new-v_old contains update/subtraction error proportional
        # to the original velocities, not just to the (possibly tiny) impulse.
        # Include that operand scale; actual numerical residuals remain retained.
        update_scale=math.fsum(self.mass[node]*(norm(new[k])+norm(old[k])) for k,node in enumerate(self.row))
        angular_scale=math.fsum(norm(s['x'][node])*self.mass[node]*(norm(new[k])+norm(old[k])) for k,node in enumerate(self.row))
        pbound=64*EPS*max(scale+update_scale,1e-30)
        lbound=128*EPS*max(angular_scale+math.fsum(norm(cross(s['x'][node],impulses[k])) for k,node in enumerate(self.row)),1e-30)
        detail={'geometry':geometry,'impulse_Ns':impulse,'closing_before_m_s':c,'closing_after_m_s':cb,'effective_inverse_mass':inv,
          'kinetic_delta_decimal_J':str(dk),'kinetic_delta_J':float(dk),'momentum_delta_Ns':dp,'angular_delta_Nms':dl,
        'momentum_rounding_bound_Ns':pbound,'angular_rounding_bound_Nms':lbound,'velocity_update_operand_scale_Ns':update_scale,'moving_owners':sum(new[k]!=old[k] for k in range(5))}
        if dk>0: raise Reject('contact-energy-positive',detail)
        if norm(dp)>pbound or norm(dl)>lbound: raise Reject('contact-nonreciprocal',detail)
        # Zero impulse may preserve closing at a separated row; active solved rows may only round near zero.
        if impulse>0 and cb < -256*EPS*max(1.,abs(c)): raise Reject('contact-closing',detail)
        return detail
    def all_areas(self,x):
        minarea=math.inf; mini=-1; mindeg=math.inf
        for i,ns in enumerate(self.body['boundary_faces']):
            a,b,c=[x[n] for n in ns]; edges=[sub(b,a),sub(c,a),sub(c,b)]
            area2=dot(cross(edges[0],edges[1]),cross(edges[0],edges[1])); edge2=max(dot(z,z) for z in edges)
            gate=128*EPS32*EPS32*edge2*edge2
            if not math.isfinite(area2) or area2<=gate: raise Reject('invalid-boundary-area',{'face':i,'nodes':ns,'cross_squared':area2,'gate':gate})
            area=.5*math.sqrt(area2)
            if area<minarea: minarea=area; mini=i
            mindeg=min(mindeg,area2/gate)
        return {'min_area_m2':minarea,'min_face':mini,'min_relative_gate_ratio':mindeg,'faces':1280}
    def full_boundary(self,x):
        a,b=[x[i] for i in self.row[:2]]; best=None
        for i,ns in enumerate(self.body['boundary_faces']):
            result=closest(a,b,[x[n] for n in ns])
            if not result['valid']: raise Reject('full-boundary-oracle-unresolved',{'face':i})
            if best is None or result['distance']<best['distance']:
                best=dict(result,face=i,nodes=ns[:])
        best['gap_m']=best['distance']-self.radius; best['faces_checked']=1280
        best['selected_row_face_matches_global_min']=best['nodes']==self.row[2:]
        if best['gap_m'] < -GAP_GATE: raise Reject('full-boundary-overlap',best)
        return best

def solve(rows,rhs):
    n=len(rows); a=[rows[i][:]+[rhs[i]] for i in range(n)]
    for k in range(n):
        p=max(range(k,n),key=lambda r:abs(a[r][k]))
        if abs(a[p][k])<1e-17: return None
        a[p],a[k]=a[k],a[p]; d=a[k][k]; a[k]=[x/d for x in a[k]]
        for r in range(n):
            if r==k: continue
            d=a[r][k]; a[r]=[a[r][j]-d*a[k][j] for j in range(n+1)]
    return [q[-1] for q in a]
def closest(a,b,t):
    # Constrained quadratic least squares, 26 bounded active subsets. No FP32 feature helper.
    A=[sub(b,a),mul(sub(t[1],t[0]),-1),mul(sub(t[2],t[0]),-1)]; r=sub(a,t[0])
    C=[[-1.,0.,0.],[1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.],[0.,1.,1.]]; d=[0.,1.,0.,0.,1.]
    best={'valid':False,'distance':math.inf}
    for mask in range(32):
        active=[i for i in range(5) if mask&(1<<i)]
        if len(active)>3: continue
        n=3+len(active); matrix=[[0.]*n for _ in range(n)]; rhs=[0.]*n
        for i in range(3):
            rhs[i]=-dot(A[i],r)
            for j in range(3): matrix[i][j]=dot(A[i],A[j])
        for q,c in enumerate(active):
            for k in range(3): matrix[k][q+3]=matrix[q+3][k]=C[c][k]
            rhs[q+3]=d[c]
        z=solve(matrix,rhs)
        if z is None or not finite(z) or any(dot(C[i],z[:3])>d[i]+64*EPS for i in range(5)): continue
        delta=add(r,sumv([mul(A[k],z[k]) for k in range(3)])); dist=norm(delta)
        if dist<best['distance']: best={'valid':True,'distance':dist,'u':z[0],'bary':[1-z[1]-z[2],z[1],z[2]],'active_mask':mask}
    return best

def state_bytes(s):
    raw=b''.join(struct.pack('<3d',*q) for q in s['x']+s['v'])
    return raw+json.dumps({k:v for k,v in s.items() if k not in ('x','v')},sort_keys=True,allow_nan=False,separators=(',',':')).encode()
def state_sha(s): return hashlib.sha256(state_bytes(s)).hexdigest()
def observables(m,s,e):
    mass=m.mass; x=s['x']; v=s['v']
    kinetic=math.fsum(.5*mass[i]*dot(v[i],v[i]) for i in range(len(mass)))
    gravity=-math.fsum(mass[i]*dot(G,x[i]) for i in range(len(mass)))
    momentum=sumv([mul(v[i],mass[i]) for i in range(len(mass))])
    angular=sumv([cross(x[i],mul(v[i],mass[i])) for i in range(len(mass))])
    return {'K_J':kinetic,'strain_J':e['U'],'gravity_J':gravity,'total_J':math.fsum([kinetic,e['U'],gravity]),'P_kg_m_s':momentum,'L_kg_m2_s':angular,
      'minJ':e['minJ'],'maxJ':e['maxJ'],'min_element':e['min_element'],'net_internal_force_N':e['net_internal_force_N'],'net_internal_torque_Nm':e['net_internal_torque_Nm']}
def kick(m,s,e,h,support,events,stage):
    for i in range(len(m.mass)):
        az=add(mul(e['force'][i],1/m.mass[i]),G)
        events['gravity_impulses'].append(mul(G,m.mass[i]*h))
        events['gravity_torques'].append(cross(s['x'][i],mul(G,m.mass[i]*h)))
        if support and s['x'][i][2]==0 and s['v'][i][2]==0 and az[2]<0:
            j=[0.,0.,-m.mass[i]*az[2]*h]; az[2]=0
            events['plane_impulses'].append(j); events['plane_torques'].append(cross(s['x'][i],j))
            events['plane_events'].append({'node':i,'stage':stage,'kind':'resting-reaction','impulse_Ns':j[2],'kinetic_loss_J':0.})
        s['v'][i]=add(s['v'][i],mul(az,h))
def remove_floor_velocity(m,s,events,i,stage):
    if s['v'][i][2]>=0: return
    old=s['v'][i][2]; j=[0.,0.,-m.mass[i]*old]; loss=.5*m.mass[i]*old*old; s['v'][i][2]=0.
    events['plane_impulses'].append(j); events['plane_torques'].append(cross(s['x'][i],j)); events['plane_loss']+=loss
    events['plane_events'].append({'node':i,'stage':stage,'kind':'incoming-normal-removal','impulse_Ns':j[2],'kinetic_loss_J':loss})

def advance(m,accepted,dt,mode='correct',fault=None):
    before=state_sha(accepted); trial=copy.deepcopy(accepted)
    events={'plane_impulses':[],'plane_torques':[],'gravity_impulses':[],'gravity_torques':[], 'projection_angular':[],'plane_loss':0.,'plane_events':[],'projections':[]}
    try:
        if not math.isfinite(dt) or dt<=0 or dt/2+dt/2!=dt: raise Reject('invalid-dt')
        initial=m.evaluate(trial['x']); start=observables(m,trial,initial)
        m.all_areas(trial['x']); m.geometry(trial['x']); initial_boundary=m.full_boundary(trial['x'])
        kick(m,trial,initial,dt/2,True,events,'first-kick')
        row1=m.impulse(trial,mode)
        for i in range(len(m.mass)): trial['x'][i]=add(trial['x'][i],mul(trial['v'][i],dt))
        free=copy.deepcopy(trial)
        free_eval=m.evaluate(free['x']); free_events={'gravity_impulses':[],'gravity_torques':[],'plane_impulses':[],'plane_torques':[],'plane_events':[]}
        kick(m,free,free_eval,dt/2,True,free_events,'free-second-kick')
        free_final=observables(m,free,free_eval)
        for i in range(len(m.mass)):
            floor=0. if i<m.n else m.radius
            if trial['x'][i][2]<floor:
                oldx=trial['x'][i][:]; trial['x'][i][2]=floor
                events['projection_angular'].append(cross(sub(trial['x'][i],oldx),mul(trial['v'][i],m.mass[i])))
                events['projections'].append({'node':i,'delta_z_m':floor-oldx[2],'source_operator':i<m.n})
                remove_floor_velocity(m,trial,events,i,'drift-projection')
        candidate_eval=m.evaluate(trial['x']); projected_Ug=-math.fsum(m.mass[i]*dot(G,trial['x'][i]) for i in range(len(m.mass)))
        potential_projection=(candidate_eval['U']+projected_Ug)-(free_eval['U']+free_final['gravity_J'])
        kick(m,trial,candidate_eval,dt/2,True,events,'second-kick')
        for i in range(len(m.mass)):
            floor=0. if i<m.n else m.radius
            if trial['x'][i][2]==floor: remove_floor_velocity(m,trial,events,i,'second-kick-removal')
        pre_row2=observables(m,trial,candidate_eval)
        row2=m.impulse(trial)
        if fault=='invalid-J':
            ns=m.elements[8768]['nodes']; trial['x'][ns[1]],trial['x'][ns[2]]=trial['x'][ns[2]],trial['x'][ns[1]]
            # This negative control reaches the constitutive inversion gate
            # directly. Other geometry gates may also fail this adversarial state.
            m.evaluate(trial['x'])
        if fault=='invalid-area':
            ns=m.row[2:]; trial['x'][ns[1]]=trial['x'][ns[0]][:]; trial['x'][ns[2]]=trial['x'][ns[0]][:]
        # Independent area gate first retains the area failure rather than masking it with J.
        areas=m.all_areas(trial['x']); geometry=m.geometry(trial['x']); final_boundary=m.full_boundary(trial['x']); final_eval=m.evaluate(trial['x']); end=observables(m,trial,final_eval)
        if not all(finite(q) for q in trial['v']): raise Reject('nonfinite-final-velocity')
        if any(q[2]<0 for q in trial['x'][:m.n]): raise Reject('floor-admission')
        d1=-row1['kinetic_delta_J']; d2=-row2['kinetic_delta_J']; loss=events['plane_loss']
        free_residual=free_final['total_J']-start['total_J']+d1
        plane_response=pre_row2['K_J']-free_final['K_J']+loss
        expected=free_residual-d1-d2-loss+potential_projection+plane_response
        energy_change=end['total_J']-start['total_J']
        gravity_work=math.fsum(m.mass[i]*dot(G,sub(trial['x'][i],accepted['x'][i])) for i in range(len(m.mass)))
        gp=sumv(events['gravity_impulses']); pp=sumv(events['plane_impulses']); gt=sumv(events['gravity_torques']); pt=sumv(events['plane_torques']); pa=sumv(events['projection_angular'])
        ledger={'dt_s':dt,'start':start,'free_finish':free_final,'pre_final_row':pre_row2,'end':end,'contact_rows':[row1,row2],
          'plane_events':events['plane_events'],'position_projections':events['projections'],'contact_dissipation_J':d1+d2,'plane_kinetic_removal_J':loss,
          'gravity_work_J':gravity_work,'position_potential_work_J':potential_projection,'plane_force_response_kinetic_J':plane_response,
          'free_split_integration_residual_J':free_residual,'physical_energy_balance_residual_J':energy_change+d1+d2+loss,
          'mechanical_minus_gravity_work_residual_J':(end['K_J']+end['strain_J'])-(start['K_J']+start['strain_J'])-gravity_work+d1+d2+loss,
          'bookkeeping_identity_error_J':energy_change-expected,'energy_closure_claim':False,
          'gravity_impulse_Ns':gp,'plane_impulse_Ns':pp,'gravity_torque_impulse_Nms':gt,'plane_torque_impulse_Nms':pt,'position_projection_angular_change_Nms':pa,
          'momentum_balance_residual_Ns':sub(sub(end['P_kg_m_s'],start['P_kg_m_s']),add(gp,pp)),
          'angular_balance_residual_Nms':sub(sub(end['L_kg_m2_s'],start['L_kg_m2_s']),sumv([gt,pt,pa])), 'final_geometry':geometry,'final_areas':areas,
          'initial_full_boundary_minimum':initial_boundary,'final_full_boundary_minimum':final_boundary}
        trial['elapsed']+=dt; trial['steps']+=1; trial['ledger'].append(ledger)
        trial['cache']={'strain_J':end['strain_J'],'minJ':end['minJ'],'last_impulse_Ns':row2['impulse_Ns'],'accepted_geometry':geometry}
        assert state_sha(accepted)==before
        accepted.clear(); accepted.update(trial)
        return {'accepted':True,'ledger':ledger,'initial_eval':initial,'final_eval':final_eval,'before_sha':before,'after_sha':state_sha(accepted)}
    except Reject as e:
        assert state_sha(accepted)==before
        return {'accepted':False,'reason':e.reason,'detail':e.detail,'before_sha':before,'after_sha':state_sha(accepted),'rollback_exact':True,'elapsed_unchanged':accepted['elapsed']}

def save_state(path,s):
    path.write_bytes(b''.join(struct.pack('<6d',*(s['x'][i]+s['v'][i])) for i in range(len(s['x']))))
def save_eval(path,e): path.write_bytes(b''.join(struct.pack('<11d',*q) for q in e['outputs']))
def baseline_control(m,s,e):
    other=ROOT/'build/elastic-yarn-common-step-independent-review/constitutive-frame38-forces.f64x3.bin'
    reference=list(struct.iter_unpack('<3d',other.read_bytes())); assert len(reference)==m.n
    differences=[norm(sub(e['force'][i],reference[i])) for i in range(m.n)]
    bodyK=math.fsum(.5*m.mass[i]*dot(s['v'][i],s['v'][i]) for i in range(m.n))
    bodyG=-math.fsum(m.mass[i]*dot(G,s['x'][i]) for i in range(m.n))
    # Independently reviewed pivoted-F^-T implementation and local gradients.
    return {'independent_force_path':str(other),'independent_force_sha256':sha(other),
      'max_force_vector_difference_N':max(differences),'witness_node':differences.index(max(differences)),
      'passed':max(differences)<1e-12 and abs(e['U']-.3856712269844007)<1e-12,
      'FEM_only_strain_J':e['U'],'FEM_only_kinetic_J':bodyK,'FEM_only_gravity_J':bodyG}
def gradient_controls(m,s):
    e=m.evaluate(s['x']); rows=[]; passed=True; h=1e-8
    for node,k in [(5,0),(1814,2),(1347,1),(7,2),(0,0),(1028,1)]:
        plus=copy.deepcopy(s['x']); minus=copy.deepcopy(s['x']); plus[node][k]+=h; minus[node][k]-=h
        derivative=(m.evaluate(plus,False)['U']-m.evaluate(minus,False)['U'])/(2*h); expected=-e['force'][node][k]
        error=abs(derivative-expected); tolerance=1e-5+1e-4*abs(expected); ok=error<=tolerance; passed&=ok
        rows.append({'node':node,'axis':k,'h_m':h,'central_energy_derivative_N':derivative,'minus_gathered_force_N':expected,'absolute_error_N':error,'tolerance_N':tolerance,'passed':ok})
    return {'passed':passed,'controls':rows,'difference_kind':'Independent central total-energy difference; full body recomputed twice per coordinate'}
def binding_control(m,s):
    actual=copy.deepcopy(s['x'])
    for i in m.row: actual[i][0]+=3000.
    exact=m.geometry(actual)
    narrowed=copy.deepcopy(actual)
    for i in m.row: narrowed[i]=[f32(x) for x in narrowed[i]]
    try: bad=m.geometry(narrowed); outcome={'accepted':True,'geometry':bad}
    except Reject as e: outcome={'accepted':False,'reason':e.reason,'detail':e.detail}
    return {'translation_m':[3000.,0.,0.],'actual_FP64_geometry':exact,'FP32_narrowed_geometry':outcome,
      'actual_positions':[actual[i] for i in m.row],'narrowed_positions':[narrowed[i] for i in m.row],'bridge_rejected':not outcome['accepted'],
      'not_an_end_to_end_production_claim':True,'radius_unchanged':m.radius}

def main():
    begin=time.time(); m=Model(); start=m.initial(); initial_sha=state_sha(start)
    print('starting full-body one-step and full-boundary coverage',flush=True)
    positive=copy.deepcopy(start); result=advance(m,positive,DT)
    print('one-step accepted='+str(result['accepted'])+' reason='+str(result.get('reason')),flush=True)
    replay=copy.deepcopy(start); repeat=advance(m,replay,DT)
    half=copy.deepcopy(start); h1=advance(m,half,DT/2); h2=advance(m,half,DT/2) if h1['accepted'] else {'accepted':False,'reason':'first-half-failed'}
    controls=[]
    for mode,fault in [('disabled-body',None),('energy-positive',None),('correct','invalid-J'),('correct','invalid-area')]:
        owned=copy.deepcopy(start); actual=advance(m,owned,DT,mode,fault); actual.pop('initial_eval',None); actual.pop('final_eval',None)
        actual['mode']=mode; actual['fault']=fault; actual['unchanged_full_state']=state_sha(owned)==initial_sha
        controls.append(actual)
    gradient=gradient_controls(m,start); binding=binding_control(m,start); baseline=baseline_control(m,start,m.evaluate(start['x']))
    artifacts={}; save_state(P/'initial.state.f64x6.bin',start)
    if result['accepted']:
        save_state(P/'one-us.state.f64x6.bin',positive); save_eval(P/'initial.piola-J-energy.f64x11.bin',result['initial_eval']); save_eval(P/'one-us.piola-J-energy.f64x11.bin',result['final_eval'])
        for name,e in [('initial',result['initial_eval']),('one-us',result['final_eval'])]:
            (P/(name+'.force.f64x3.bin')).write_bytes(b''.join(struct.pack('<3d',*q) for q in e['force']))
        (P/'one-us.ledger.json').write_text(json.dumps(result['ledger'],indent=2,allow_nan=False)+'\n')
    if h2['accepted']:
        save_state(P/'two-half-us.state.f64x6.bin',half); (P/'two-half-us.ledgers.json').write_text(json.dumps(half['ledger'],indent=2,allow_nan=False)+'\n')
    comparison=None
    if result['accepted'] and h2['accepted']:
        comparison={'same_total_interval_s':DT,'one_step_elapsed_s':positive['elapsed'],'two_step_elapsed_s':half['elapsed'],
          'max_position_difference_m':max(norm(sub(a,b)) for a,b in zip(positive['x'],half['x'])),
          'max_velocity_difference_m_s':max(norm(sub(a,b)) for a,b in zip(positive['v'],half['v'])),
          'energy_difference_J':half['ledger'][-1]['end']['total_J']-positive['ledger'][-1]['end']['total_J'],
          'contact_impulse_one_Ns':sum(q['impulse_Ns'] for q in positive['ledger'][0]['contact_rows']),
          'contact_impulse_two_Ns':sum(q['impulse_Ns'] for e in half['ledger'] for q in e['contact_rows']),
          'physical_balance_residual_one_J':positive['ledger'][0]['physical_energy_balance_residual_J'],
          'physical_balance_residual_two_J':sum(e['physical_energy_balance_residual_J'] for e in half['ledger']),
          'bounded_observation_only_not_temporal_convergence':True}
    result.pop('initial_eval',None); result.pop('final_eval',None); repeat.pop('initial_eval',None); repeat.pop('final_eval',None)
    exact=result==repeat and state_bytes(positive)==state_bytes(replay)
    passed=result['accepted'] and h2['accepted'] and exact and gradient['passed'] and baseline['passed'] and binding['bridge_rejected'] and all(not q['accepted'] and q['rollback_exact'] and q['unchanged_full_state'] for q in controls)
    evidence={'schema':'isolated-FP64-common-yarn-FEM-step-v1','passed_bounded_controls':passed,'physics_source':m.body['physics_source_revision'],
      'source_script_sha256':sha(pathlib.Path(__file__)),'input_sha256':m.inputs,'new_discretization':'FP64 source constitutive law and support-Verlet operator with reciprocal instantaneous contact rows before drift and after second kick',
      'body_nodes':m.n,'authored_yarn_nodes':2,'tetrahedra':len(m.elements),'all_node_masses_positive':True,'total_FEM_mass_kg':math.fsum(m.mass[:m.n]),
      'dt_s':DT,'radius_m_exact_represented':m.radius,'native_J_gate':J_GATE,'native_rest_consistency_gate':REST_GATE,'contact_accuracy_m':GAP_GATE,'max_rest_consistency_error':m.max_rest_error,
      'initial_state_sha256':initial_sha,'one_step':result,'exact_replay':exact,'half_steps_accepted':[h1['accepted'],h2['accepted']],
      'matched_interval':comparison,'energy_gradient_controls':gradient,'independent_constitutive_baseline':baseline,'negative_controls':controls,'geometry_binding_control':binding,
      'boundaries':['No native byte exact continuation: inverse-rest is owning-source reconstruction; full frame38 historical native checkpoint absent.',
        'The complete FEM body responds, but the yarn is a local authored two-node segment with no internal yarn stress, friction, full bag or full scene.',
        'Startup approximately 1 micrometre overlap is within the unchanged 2 micrometre target. This is not a moving CCD or no-tunnelling proof.',
        'Energy residual is retained as measured; bookkeeping identity is not physical energy closure.',
        'One 1us step vs two 0.5us steps is a bounded numerical observation, not temporal convergence.'],
      'runtime_s':time.time()-begin}
    (P/'evidence.json').write_text(json.dumps(evidence,indent=2,allow_nan=False)+'\n')
    for path in P.iterdir():
        if path.is_file() and path.name not in ['receipt.json','run.log','actual.exit']: artifacts[path.name]={'sha256':sha(path),'bytes':path.stat().st_size}
    (P/'receipt.json').write_text(json.dumps({'actual_program_exit':0 if passed else 1,'artifacts':artifacts,'inputs':m.inputs,'interpreter':sys.executable,'python_version':sys.version,'source_sha256':sha(pathlib.Path(__file__))},indent=2)+'\n')
    print(json.dumps({'passed':passed,'one_step_accepted':result['accepted'],'one_step_reason':result.get('reason'),'exact_replay':exact,'half_steps':[h1['accepted'],h2['accepted']],
       'negatives':[(q['mode'],q['fault'],q.get('reason')) for q in controls],'gradient_pass':gradient['passed'],'binding_rejected':binding['bridge_rejected'],'runtime_s':time.time()-begin},indent=2))
    return 0 if passed else 1
if __name__=='__main__': sys.exit(main())
