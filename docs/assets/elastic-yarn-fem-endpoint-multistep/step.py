#!/usr/bin/env python3
"""New shared two-row FP64 normal block in the immutable full FEM split.
The copied centroid source supplies only owning inputs/constitutive/split helpers.
"""
import copy,hashlib,importlib.util,json,math,pathlib,struct,sys,time
from decimal import Decimal as D,localcontext
P=pathlib.Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('frozen_centroid',P/'centroid_step.py')
B=importlib.util.module_from_spec(spec);spec.loader.exec_module(B)
BASE_SHA='b249e25faaeba30e486e8c2b70d76ab875b69d3a716518f9a62858a6c5b4379b'
assert B.sha(P/'centroid_step.py')==BASE_SHA
add,sub,mul,dot,cross,norm,sumv=B.add,B.sub,B.mul,B.dot,B.cross,B.norm,B.sumv
Reject=B.Reject; EPS=B.EPS

class Model(B.Model):
    def __init__(self):
        super().__init__()
        self.inputs[str(P/'centroid_step.py')]=BASE_SHA
        self.inputs[str(pathlib.Path(__file__))]=B.sha(pathlib.Path(__file__))
    def endpoint_rows(self,x):
        own=[x[i] for i in self.row]; a,b,*tri=own
        e0=sub(tri[1],tri[0]); e1=sub(tri[2],tri[0]); aa=dot(e0,e0); ab=dot(e0,e1); bb=dot(e1,e1)
        determinant=aa*bb-ab*ab
        if not math.isfinite(determinant) or determinant<=128*EPS32_SQUARED*max(aa,bb)**2:
            raise Reject('endpoint-projection-degenerate',{'gram_det':determinant})
        rows=[]
        for endpoint,yarn in enumerate([a,b]):
            r=sub(yarn,tri[0]); d0=dot(r,e0); d1=dot(r,e1)
            beta1=(d0*bb-d1*ab)/determinant; beta2=(d1*aa-d0*ab)/determinant; beta=[1-beta1-beta2,beta1,beta2]
            # Only the independently stated interior manifold is implemented.
            if not B.finite(beta) or min(beta)<=64*EPS or max(beta)>=1-64*EPS:
                raise Reject('unsupported-endpoint-feature',{'endpoint':endpoint,'bary':beta})
            tp=sumv([mul(tri[k],beta[k]) for k in range(3)]); delta=sub(yarn,tp); distance=norm(delta)
            normal_cutoff=8*B.EPS32*max(1.,norm(yarn),norm(tp))
            if distance<=normal_cutoff:raise Reject('undefined-endpoint-normal',{'endpoint':endpoint,'distance':distance})
            oracle=B.closest(yarn,yarn,tri)
            if not oracle['valid'] or abs(distance-oracle['distance'])>B.GAP_GATE:
                raise Reject('endpoint-manifold-unresolved',{'endpoint':endpoint,'distance':distance,'oracle':oracle})
            gap=distance-self.radius
            if gap < -B.GAP_GATE:raise Reject('endpoint-overlap',{'endpoint':endpoint,'gap':gap})
            if gap>B.GAP_GATE:raise Reject('unsupported-separated-endpoint',{'endpoint':endpoint,'gap':gap})
            weight=[float(endpoint==0),float(endpoint==1)]+[-q for q in beta]
            rows.append({'endpoint':endpoint,'u':float(endpoint),'bary':beta,'normal':mul(delta,1/distance),
              'distance':distance,'gap':gap,'weight':weight,'yarn_point':yarn[:],'triangle_point':tp})
        if dot(rows[0]['normal'],rows[1]['normal'])<1-512*EPS:raise Reject('unsupported-normal-orientation')
        return rows
    def geometry(self,x):
        result=super().geometry(x);result['endpoint_rows']=self.endpoint_rows(x);return result
    def impulse(self,s,mode='correct'):
        geometry=self.geometry(s['x']); rows=geometry['endpoint_rows']; grad=[[mul(r['normal'],w) for w in r['weight']] for r in rows]
        q=[math.fsum(dot(g[k],s['v'][node]) for k,node in enumerate(self.row)) for g in grad]
        W=[[math.fsum(dot(grad[i][k],grad[j][k])/self.mass[node] for k,node in enumerate(self.row)) for j in range(2)] for i in range(2)]
        determinant=W[0][0]*W[1][1]-W[0][1]*W[1][0]
        if not B.finite(sum(W,[])+q) or W[0][0]<=0 or W[1][1]<=0 or determinant<=64*EPS*W[0][0]*W[1][1]:raise Reject('normal-block-singular',{'W':W})
        velocity_scale=max(1.,*(norm(s['v'][i]) for i in self.row)); tolerance=256*EPS*velocity_scale
        candidates=[]
        for mask in range(4):
            if mask==0:J=[0.,0.]
            elif mask==1:J=[-q[0]/W[0][0],0.]
            elif mask==2:J=[0.,-q[1]/W[1][1]]
            else:J=[(-q[0]*W[1][1]+q[1]*W[0][1])/determinant,(-q[1]*W[0][0]+q[0]*W[1][0])/determinant]
            if not B.finite(J) or min(J)<0:continue
            post=[q[i]+math.fsum(W[i][j]*J[j] for j in range(2)) for i in range(2)]
            if min(post)<-tolerance or any(mask&(1<<i) and abs(post[i])>tolerance for i in range(2)):continue
            analytic_delta=dot(q,J)+.5*math.fsum(J[i]*W[i][j]*J[j] for i in range(2) for j in range(2))
            candidates.append((analytic_delta,mask,J,post))
        if not candidates:raise Reject('normal-block-LCP-unresolved',{'q':q,'W':W})
        analytic,active,J,expected_post=min(candidates,key=lambda t:t[0]);J=J[:]
        if mode=='energy-positive':J=[3*j for j in J]
        applied_analytic=dot(q,J)+.5*math.fsum(J[i]*W[i][j]*J[j] for i in range(2) for j in range(2))
        old=[s['v'][i][:] for i in self.row]
        # Each owner receives the SUM of the coupled row impulses exactly once.
        for k,node in enumerate(self.row):
            if mode=='disabled-body' and k>=2:continue
            joint=sumv([mul(grad[i][k],J[i]) for i in range(2)])
            s['v'][node]=add(s['v'][node],mul(joint,1/self.mass[node]))
        new=[s['v'][i][:] for i in self.row]
        with localcontext() as ctx:
            ctx.prec=80
            dk=sum(D.from_float(self.mass[node])/2*sum((D.from_float(new[k][j])-D.from_float(old[k][j]))*(D.from_float(new[k][j])+D.from_float(old[k][j])) for j in range(3)) for k,node in enumerate(self.row))
        actual_imp=[mul(sub(new[k],old[k]),self.mass[node]) for k,node in enumerate(self.row)]
        dp=sumv(actual_imp);dl=sumv([cross(s['x'][node],actual_imp[k]) for k,node in enumerate(self.row)])
        update_scale=math.fsum(self.mass[node]*(norm(new[k])+norm(old[k])) for k,node in enumerate(self.row))
        imp_scale=math.fsum(norm(z) for z in actual_imp)
        angular_scale=math.fsum(norm(s['x'][node])*self.mass[node]*(norm(new[k])+norm(old[k])) for k,node in enumerate(self.row))
        pbound=64*EPS*max(update_scale+imp_scale,1e-30)
        lbound=128*EPS*max(angular_scale+math.fsum(norm(cross(s['x'][node],actual_imp[k])) for k,node in enumerate(self.row)),1e-30)
        postq=[math.fsum(dot(g[k],new[k]) for k in range(5)) for g in grad]
        detail={'geometry':geometry,'normal_block':True,'row_impulses_Ns':J,'impulse_Ns':math.fsum(J),'LCP_active_mask':active,
          'closing_before_rows_m_s':q,'closing_after_rows_m_s':postq,'expected_after_rows_m_s':expected_post,'W_inverse_mass':W,
          'selected_LCP_kinetic_delta_J':analytic,'analytic_kinetic_delta_J':applied_analytic,'kinetic_delta_decimal_J':str(dk),'kinetic_delta_J':float(dk),
          'momentum_delta_Ns':dp,'angular_delta_Nms':dl,'momentum_rounding_bound_Ns':pbound,'angular_rounding_bound_Nms':lbound,
          'velocity_rounding_bound_m_s':tolerance,'moving_owners':sum(new[k]!=old[k] for k in range(5)),
          'old_five_velocities':old,'new_five_velocities':new}
        if dk>0:raise Reject('contact-energy-positive',detail)
        if norm(dp)>pbound or norm(dl)>lbound:raise Reject('contact-nonreciprocal',detail)
        if min(postq)<-tolerance:raise Reject('normal-block-actual-closing',detail)
        return detail

EPS32_SQUARED=B.EPS32*B.EPS32
def block_controls(m,start):
    controls=[];n=m.endpoint_rows(start['x'])[0]['normal']
    for q,expected in [([1.,1.],0),([-1.,.1],1),([.1,-1.],2),([-1.,-1.],3)]:
        s=copy.deepcopy(start)
        for node in m.row:s['v'][node]=[0.,0.,0.]
        for k in range(2):s['v'][m.row[k]]=mul(n,q[k])
        try:
            out=m.impulse(s);passed=out['LCP_active_mask']==expected and out['kinetic_delta_J']<=0
            controls.append({'input_q':q,'expected_mask':expected,'passed':passed,'result':out})
        except Reject as e:controls.append({'input_q':q,'expected_mask':expected,'passed':False,'reason':e.reason,'detail':e.detail})
    moved=copy.deepcopy(start);moved['x'][m.row[0]][0]+=.05
    try:m.endpoint_rows(moved['x']);feature={'rejected':False}
    except Reject as e:feature={'rejected':e.reason=='unsupported-endpoint-feature','reason':e.reason,'detail':e.detail}
    return {'passed':all(z['passed'] for z in controls) and feature['rejected'],'four_active_sets':controls,'unsupported_edge_feature':feature,
      'scope':'Direct shared-block algebra controls with altered velocities; not additional full-body trajectories'}

def velocity_scan(m,s,enforce=False):
    a,b=[s['x'][i] for i in m.row[:2]]; best=None; contacts=[]
    for face,ns in enumerate(m.body['boundary_faces']):
        tri=[s['x'][i] for i in ns]; oracle=B.closest(a,b,tri)
        if not oracle['valid']:raise Reject('global-velocity-oracle-unresolved',{'face':face})
        u=oracle['u'];beta=oracle['bary'];yp=add(mul(a,1-u),mul(b,u));tp=sumv([mul(tri[k],beta[k]) for k in range(3)])
        delta=sub(yp,tp);distance=norm(delta);gap=distance-m.radius
        entry=dict(oracle,face=face,nodes=ns[:],gap_m=gap)
        if distance>8*B.EPS32*max(1.,norm(yp),norm(tp)):
            n=mul(delta,1/distance);yv=add(mul(s['v'][m.row[0]],1-u),mul(s['v'][m.row[1]],u));tv=sumv([mul(s['v'][ns[k]],beta[k]) for k in range(3)])
            entry['closing_m_s']=dot(n,sub(yv,tv));entry['normal']=n
        elif gap<=B.GAP_GATE:raise Reject('global-velocity-normal-unresolved',entry)
        if best is None or distance<best['distance']:best=entry
        if gap<=B.GAP_GATE:
            contacts.append(entry)
            if ns!=m.row[2:]:raise Reject('unsupported-other-boundary-contact',entry)
            tol=512*EPS*max(1.,*(norm(s['v'][i]) for i in m.row))
            if enforce and entry['closing_m_s'] < -tol:raise Reject('global-normal-manifold-closing',entry)
    return {'faces_checked':1280,'minimum':best,'contacts_within_2um':contacts,'final_velocity_enforced':enforce,'numerical_oracle_not_CCD':True}

def advance(m,accepted,dt,mode='correct',fault=None):
    before=B.state_sha(accepted);trial=copy.deepcopy(accepted)
    try:
        initial_scan=velocity_scan(m,trial)
        result=B.advance(m,trial,dt,mode,fault)
        if not result['accepted']:
            assert B.state_sha(accepted)==before;return result
        final_scan=velocity_scan(m,trial,True)
        ledger=result['ledger'];ledger['initial_global_velocity_scan']=initial_scan;ledger['final_global_velocity_scan']=final_scan
        ledger['normal_manifold_operator']='Two endpoint-to-interior-triangle rows, shared2x2 nonnegative block; no friction/CCD'
        trial['cache']['accepted_global_velocity_scan']=final_scan
        assert B.state_sha(accepted)==before
        accepted.clear();accepted.update(trial);result['after_sha']=B.state_sha(accepted)
        return result
    except Reject as e:
        assert B.state_sha(accepted)==before
        return {'accepted':False,'reason':e.reason,'detail':e.detail,'before_sha':before,'after_sha':B.state_sha(accepted),'rollback_exact':True,'elapsed_unchanged':accepted['elapsed']}

def main():
    begin=time.time();m=Model();start=m.initial();print('starting shared normal-block full-body step',flush=True)
    owner=copy.deepcopy(start);r=advance(m,owner,B.DT);print('one-step '+str(r['accepted'])+' reason='+str(r.get('reason')),flush=True)
    repeat=copy.deepcopy(start);rp=advance(m,repeat,B.DT);exact=r==rp and B.state_bytes(owner)==B.state_bytes(repeat)
    half=copy.deepcopy(start);h1=advance(m,half,B.DT/2);h2=advance(m,half,B.DT/2) if h1['accepted'] else {'accepted':False,'reason':'first-half-failed'}
    controls=[]
    for mode,fault in [('disabled-body',None),('energy-positive',None),('correct','invalid-J'),('correct','invalid-area')]:
        negative=copy.deepcopy(start);out=advance(m,negative,B.DT,mode,fault);out.pop('initial_eval',None);out.pop('final_eval',None)
        out.update(mode=mode,fault=fault,unchanged_full_state=B.state_sha(negative)==B.state_sha(start));controls.append(out)
    gradient=B.gradient_controls(m,start);binding=B.binding_control(m,start);baseline=B.baseline_control(m,start,m.evaluate(start['x']));algebra=block_controls(m,start)
    B.save_state(P/'initial.state.f64x6.bin',start)
    if r['accepted']:
        B.save_state(P/'one-us.state.f64x6.bin',owner)
        for name,e in [('initial',r['initial_eval']),('one-us',r['final_eval'])]:
            B.save_eval(P/(name+'.piola-J-energy.f64x11.bin'),e)
            (P/(name+'.force.f64x3.bin')).write_bytes(b''.join(struct.pack('<3d',*q) for q in e['force']))
        (P/'one-us.ledger.json').write_text(json.dumps(r['ledger'],indent=2,allow_nan=False)+'\n')
    if h2['accepted']:
        B.save_state(P/'two-half-us.state.f64x6.bin',half);(P/'two-half-us.ledgers.json').write_text(json.dumps(half['ledger'],indent=2,allow_nan=False)+'\n')
    comparison=None
    if r['accepted'] and h2['accepted']:
        comparison={'interval_s':B.DT,'one_elapsed_s':owner['elapsed'],'two_elapsed_s':half['elapsed'],
          'max_position_difference_m':max(norm(sub(a,b)) for a,b in zip(owner['x'],half['x'])),
          'max_velocity_difference_m_s':max(norm(sub(a,b)) for a,b in zip(owner['v'],half['v'])),
          'energy_difference_J':half['ledger'][-1]['end']['total_J']-owner['ledger'][-1]['end']['total_J'],
          'balance_residual_one_J':owner['ledger'][0]['physical_energy_balance_residual_J'],
          'balance_residual_two_J':math.fsum(q['physical_energy_balance_residual_J'] for q in half['ledger']),
          'not_temporal_convergence':True}
    for out in [r,rp]:out.pop('initial_eval',None);out.pop('final_eval',None)
    centroid_dir=B.ROOT/'build/elastic-yarn-fem-common-step';centroid=json.loads((centroid_dir/'evidence.json').read_text())
    centroid_state=copy.deepcopy(start);raw=list(struct.iter_unpack('<6d',(centroid_dir/'one-us.state.f64x6.bin').read_bytes()))
    centroid_state['x']=[list(z[:3]) for z in raw];centroid_state['v']=[list(z[3:]) for z in raw]
    centroid_global=velocity_scan(m,centroid_state)
    try:velocity_scan(m,centroid_state,True);centroid_negative=False
    except Reject as e:centroid_negative=e.reason=='global-normal-manifold-closing'
    passed=r['accepted'] and h2['accepted'] and exact and centroid_negative and gradient['passed'] and baseline['passed'] and binding['bridge_rejected'] and algebra['passed'] and all(not q['accepted'] and q['unchanged_full_state'] and q['rollback_exact'] for q in controls)
    evidence={'schema':'numi.elastic-yarn.complete-FEM-shared-normal-block-step.v1','passed_bounded_controls':passed,'source_sha256':B.sha(pathlib.Path(__file__)),
      'copied_split_source_sha256':BASE_SHA,'input_sha256':m.inputs,'physics_source_revision':m.body['physics_source_revision'],'dt_s':B.DT,'body_nodes':m.n,'tetrahedra':10240,
      'one_step':r,'exact_replay':exact,'half_steps_accepted':[h1['accepted'],h2['accepted']],'matched_interval':comparison,'negative_controls':controls,
      'baseline':baseline,'energy_gradient_controls':gradient,'geometry_binding_control':binding,'block_algebra_controls':algebra,
      'centroid_diagnostic_source_sha256':BASE_SHA,'centroid_diagnostic_evidence_sha256':B.sha(centroid_dir/'evidence.json'),
      'centroid_diagnostic_global_velocity':centroid_global,'centroid_full_manifold_negative_reproduced':centroid_negative,
      'limits':['New FP64 split and normal2x2 block; not native checkpoint replay.','Only endpoint projections strictly inside the selected face and no other active boundary face are implemented; feature changes reject.',
        'All1280 initial/final distances and velocities are numerical FP64 active-set evidence; not exact-real lower certificates or CCD.',
        'The local yarn has two authored nodes and no internal yarn elasticity/friction/fullbag.','Initialwithin2um overlap is retained; whole-interval no-tunnelling is open.',
        'Energy residual is measured and retained; no energy closure or full temporal convergence claim.','Historical inverse-upload/full native frame38 checkpoint absent; generic yarn-floor resting remains unqualified.'],
      'runtime_s':time.time()-begin}
    (P/'evidence.json').write_text(json.dumps(evidence,indent=2,allow_nan=False)+'\n')
    (P/'receipt.json').write_text(json.dumps({'actual_program_exit':0 if passed else 1,'source_sha256':evidence['source_sha256'],'copied_split_source_sha256':BASE_SHA,
      'input_sha256':m.inputs,'artifacts':{q.name:{'sha256':B.sha(q),'bytes':q.stat().st_size} for q in P.iterdir() if q.is_file() and q.name not in ['receipt.json','run.log','actual.exit']}},indent=2)+'\n')
    print(json.dumps({'passed':passed,'accepted':r['accepted'],'reason':r.get('reason'),'exact_replay':exact,'half_steps':[h1['accepted'],h2['accepted']],
      'centroid_manifold_negative':centroid_negative,'negatives':[q.get('reason') for q in controls],'runtime_s':time.time()-begin},indent=2))
    return 0 if passed else 1
if __name__=='__main__':sys.exit(main())
