#!/usr/bin/env python3
"""Read-only source-bound predicate and retained-triangle audit; no simulation."""
import csv
import hashlib
import json
import math
import pathlib
import re
from decimal import Decimal as D, localcontext

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = pathlib.Path(__file__).resolve().parent
FROZEN = ROOT / 'build/loaded-drop-investigation-corrected-launch'
STATUS = ROOT / 'build/loaded-drop-investigation-terminal-audit/status.json'
LOG = ROOT / 'build/loaded-drop-investigation-corrected-96.log'
EXIT = ROOT / 'build/loaded-drop-investigation-corrected-96.exit'
GATE = D('1e-8')

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def norm(v):
    return sum(x*x for x in v).sqrt()

def sub(a,b):
    return tuple(x-y for x,y in zip(a,b))

def cross(a,b):
    return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])

def uncertainty(x):
    # defaultfloat precision9 means nine significant digits, including omitted
    # trailing zeros. Add two FP64 ulps beyond half a decimal quantum.
    if not x:
        return D(0)  # A finite FP64 nonzero number is not formatted as literal0.
    return D(5).scaleb(x.adjusted()-9) + D.from_float(2*math.ulp(float(x)))

def area_bound(a,b,c):
    u,w=sub(b,a),sub(c,a)
    ea,eb,ec=[tuple(uncertainty(x) for x in p) for p in (a,b,c)]
    du=norm(tuple(x+y for x,y in zip(ea,eb))).next_plus()
    dw=norm(tuple(x+y for x,y in zip(ea,ec))).next_plus()
    error=((norm(u).next_plus()*dw + norm(w).next_plus()*du + du*dw)/2).next_plus()
    area=norm(cross(u,w))/2
    return area,max(D(0),(area-error).next_minus()),(area+error).next_plus(),error

def parse_obj(p):
    vertices=[]; faces=[]; metadata=[]
    for line in p.read_text().splitlines():
        if line.startswith('v '):
            vertices.append(tuple(D(x) for x in line.split()[1:4]))
        elif line.startswith('f '):
            faces.append(tuple(int(x.split('/')[0])-1 for x in line.split()[1:]))
        elif line.startswith('# grip ') or line.startswith('# ball '):
            metadata.append(line)
    assert len(vertices)==1465 and len(faces)==2880
    assert all(len(f)==3 and all(0<=i<len(vertices) for i in f) for f in faces)
    assert all(x.is_finite() for v in vertices for x in v)
    return vertices,faces,metadata

def numeric_gates(text):
    values=dict(re.findall(r'\b([A-Za-z_][A-Za-z_0-9]*)=([^\s]+)', text))
    rules=[
        ('escaped_mask','==','0'),('spilled_mask','==','0'),
        ('max_ground_contact_correction','<','0.004001'),
        ('max_regrab_capture_distance','<=','0.120000000001'),
        ('max_regrab_capture_error','<=','1e-9'),
        ('min_triangle_area','>','1e-8'),
        ('max_warp_extension','<','0.30'),('max_warp_compression','<','0.60'),
        ('max_weft_extension','<','0.30'),('max_weft_compression','<','0.60'),
        ('max_bottom_extension','<','0.30'),('max_bottom_compression','<','0.60'),
        ('max_knot_angle_error','<','0.80'),('max_ball_contact_correction','<','0.010'),
        ('max_published_ball_penetration','<','2e-6'),
        ('max_published_fruit_pair_penetration','<','2e-6'),
        ('max_published_ground_penetration','<','2e-6'),
        ('max_published_primitive_self_penetration','<','2e-6'),
        ('max_published_local_node_penetration','<','2e-6'),
        ('max_published_strain_limit_violation','<','2e-6'),
        ('max_self_penetration','<','0.008'),
        ('final_primitive_self_penetration','<','2e-6'),
        ('final_strain_limit_violation','<','2e-6'),
        ('max_speed','<','30'),('max_angular_speed','<','200'),
        ('max_grip_force','<','500'),
        ('max_friction_cone_ratio','<=','1.000000000001'),
        ('max_rolling_resistance_ratio','<=','1.000000000001')]
    ops={'<':lambda a,b:a<b,'>':lambda a,b:a>b,'<=':lambda a,b:a<=b,'==':lambda a,b:a==b}
    result=[]
    for key,op,target in rules:
        token=values[key];value=D(token);threshold=D(target)
        ok=ops[op](value,threshold)
        # Final summary uses fixed precision9. Integer masks are exact.
        interval=(value,value) if op=='==' else (max(D(0),value-D('5e-10')),value+D('5e-10'))
        robust=(ops[op](interval[0],threshold) and ops[op](interval[1],threshold))
        robust_fail=(not ops[op](interval[0],threshold) and not ops[op](interval[1],threshold))
        result.append({'log_field':key,'value':token,'comparison':op,'unchanged_threshold':target,
            'rounded_value_pass':ok,'rounding_robust_status': 'PASS' if robust else 'FAIL' if robust_fail else 'INDETERMINATE',
            'possible_underlying_value_interval':[str(x) for x in interval]})
    other=[
        {'predicate':'deterministic','status':'PASS_REPORTED','evidence':values['deterministic']},
        {'predicate':'pickupOutcome','status':'PASS_NOT_APPLICABLE','evidence':'scenario=recorded'},
        {'predicate':'spatialPatchValid','status':'PASS_REPORTED','evidence':'patch_selection_count='+values['patch_selection_count']+'; zero bypasses patch topology branch'},
        {'predicate':'cloth mass','status':'PASS_SOURCE_INVARIANT','evidence':'1369 ordinary nodes at5e-5kg and96 hem nodes at1e-4kg; mass unchanged by simulate'},
        {'predicate':'allFinite including final grip/fruit unit quaternion checks','status':'PARTIAL_OBSERVABILITY',
         'evidence':'Saved positions/OBJ fruit velocity/quaternions finite; final cloth velocities are not exported; OBJ quaternion precision9 cannot generally certify1e-9 norm gate.'}]
    assert values['deterministic']=='true' and values['scenario']=='recorded' and values['patch_selection_count']=='0'
    return result,other,values

def main():
    manifest=json.loads((FROZEN/'manifest.json').read_text())
    for rel,expected in manifest['sha256'].items():
        assert sha(ROOT/rel)==expected,(rel,'frozen hash mismatch')
    status=json.loads(STATUS.read_text())
    case=status['cases']['96']
    assert status['status']=='TERMINAL' and case['actual_solver_exit']==1
    immutable=case['immutable_input_sha256']
    paths=sorted(pathlib.Path(p) for p in immutable if p.endswith('.obj'))
    assert len(paths)==76
    for p in paths:
        assert sha(p)==immutable[str(p)],(p,'retained state changed')
    assert EXIT.read_text().strip()=='1'
    rows=[];overall=None;topology=None;checks=0;max_fp64_error=D(0)
    with localcontext() as ctx:
        ctx.prec=60
        for p in paths:
            v,f,meta=parse_obj(p)
            if topology is None:topology=f
            assert f==topology
            minimum=None;below=[]
            for i,face in enumerate(f):
                a,b,c=[v[n] for n in face]
                area=norm(cross(sub(b,a),sub(c,a)))/2
                fu=tuple(float(b[k])-float(a[k]) for k in range(3));fw=tuple(float(c[k])-float(a[k]) for k in range(3))
                float_area=.5*math.hypot(*cross(fu,fw))
                max_fp64_error=max(max_fp64_error,abs(D.from_float(float_area)-area))
                checks+=1
                if minimum is None or area<minimum[0]:minimum=(area,i,face,a,b,c)
                if area<=GATE:below.append(i)
            area,i,face,a,b,c=minimum
            _,lo,hi,error=area_bound(a,b,c)
            # Independent length-only Gram/Heron identity on each minimum.
            u,w=sub(b,a),sub(c,a)
            uu=sum(x*x for x in u);ww=sum(x*x for x in w);uw=sum(x*y for x,y in zip(u,w))
            gram=max(D(0),uu*ww-uw*uw).sqrt()/2
            assert abs(gram-area)<D('1e-45')
            row={'path':str(p),'sha256':sha(p),'minimum_serialized_area_m2':str(area),
                'triangle_zero_based':i,'nodes_zero_based':list(face),
                'positions_m':[[str(x) for x in pos] for pos in (a,b,c)],
                'possible_unserialized_witness_area_m2':[str(lo),str(hi)],
                'serialization_area_error_bound_m2':str(error),
                'minimum_gram_area_m2':str(gram),'triangle_count':len(f),
                'serialized_faces_at_or_below_gate':below,
                'witness_robustly_fails_gate':hi<=GATE}
            rows.append(row)
            if overall is None or area<D(overall['minimum_serialized_area_m2']):overall=row
    gates,other,values=numeric_gates(LOG.read_text())
    definite=[g['log_field'] for g in gates if g['rounding_robust_status']=='FAIL']
    rounded_failed=[g['log_field'] for g in gates if not g['rounded_value_pass']]
    assert definite==['min_triangle_area'] and rounded_failed==['min_triangle_area']
    assert overall['witness_robustly_fails_gate']
    failing=[r for r in rows if r['witness_robustly_fails_gate']]
    ordinary=5e-5;hem=1e-4;expected=1369*ordinary+96*hem
    sequential=sum([ordinary]*(48*26)+[hem]*(48*2)+[ordinary]*121)
    mass_error=abs(sequential-expected)
    assert mass_error<1e-12
    result={'schema':'numi.loaded-drop.cpu96.retained-render-triangle-area-attribution.v1',
        'source_parent_commit':manifest['physics_parent_commit'],
        'all_seven_frozen_physics_hashes_verified':True,'source_sha256':sha(FROZEN/'cloth_bag.cpp'),
        'source_manifest_sha256':sha(FROZEN/'manifest.json'),'terminal_status_sha256':sha(STATUS),
        'terminal_log_sha256':sha(LOG),'actual_solver_exit':1,'actual_exit_file_sha256':sha(EXIT),
        'audit_script_sha256':sha(pathlib.Path(__file__).resolve()),
        'method':'Independent Decimal60 on every serialized triangle; FP64 cross comparison; independent Decimal60 Gram identity for each saved-state minimum; perturbation norm bound for precision9 coordinate serialization.',
        'retained_state_count':len(rows),'triangle_evaluations':checks,
        'maximum_fp64_vs_decimal_area_difference_m2':str(max_fp64_error),
        'unchanged_area_gate':{'predicate':'minimumTriangleArea > 1e-8','threshold_m2':str(GATE)},
        'reported_minimum_triangle_area_m2':values['min_triangle_area'],
        'reported_metric_possible_interval_m2':['5.5e-9','6.5e-9'],
        'metric_semantics':{'source_area_update_lines':[4954,4961],'source_frame_end_call_line':6024,
            'substep_loop_ends_line':5992,'updated_at':'each of720 frame ends, after all96 substeps; not each substep',
            'minimum_witness_instrumented':False,'saved_regular_cadence_frames':10},
        'overall_saved_minimum':overall,
        'saved_states_with_robust_area_failure_count':len(failing),
        'saved_states_with_robust_area_failure':[r['path'] for r in failing],
        'logged_numeric_gate_comparison':gates,'remaining_acceptance_predicates':other,
        'source_cone_invariants':{
            'friction':'All five frozen recordFrictionImpulse call sites pass std::min(requiredImpulse,frictionLimit) and the same positive frictionLimit. For represented finite values their quotient cannot exceed1; std::max also cannot promote a NaN quotient. Source lines2143–2157,2230–2244,2279–2295,3656–3673,4471–4491.',
            'rolling':'angularImpulse=std::min(requiredAngularImpulse,rollingImpulseLimit); the recorded ratio divides by that same positive rollingImpulseLimit. Source lines2180–2197.',
            'boundary':'Source algebra explains the rounded unit ratios; this is not an independent retained per-contact impulse/work ledger.'},
        'source_mass_check':{'sequential_mass_kg':sequential,'kClothMass_kg':expected,'error_kg':mass_error,'unchanged_gate_kg':1e-12},
        'sole_rounded_logged_numeric_failure':'min_triangle_area',
        'sole_rounding_robust_logged_numeric_failure':'min_triangle_area',
        'unresolved_predicate_observability':[g['log_field'] for g in gates if g['rounding_robust_status']=='INDETERMINATE']+['final cloth velocity finiteness','exact final unit-quaternion norms'],
        'rendered_witness_cell':{'bottom_grid_row':9,'bottom_grid_column':2,'triangle_in_cell':1,'topology':'(a,c,d) on odd(row+column) cell','source_lines':[949,965]},
        'all76_retained_contact_geometry_pass':case['all_complete_saved_geometry_pass'],
        'boundary':'Retained triangle area failure is independently confirmed. This is the sole definite numeric failure visible in the rounded log, not proof every unexported runtime predicate passed. The exact all720-frame area minimum witness and all substep minima are not retained; no claim of absent transient failures, inversion, solver-operator attribution or physical energy closure. No simulation or GPU work.',
        'states':rows}
    (OUT/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    with (OUT/'states.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['path','minimum_area_m2','triangle_zero_based','nodes_zero_based','unserialized_lower_m2','unserialized_upper_m2','robust_failure','faces_below_gate'])
        for r in rows:writer.writerow([r['path'],r['minimum_serialized_area_m2'],r['triangle_zero_based'],','.join(map(str,r['nodes_zero_based'])),*r['possible_unserialized_witness_area_m2'],r['witness_robustly_fails_gate'],len(r['serialized_faces_at_or_below_gate'])])
    # Assert inputs remain identical after analysis.
    for p in paths:assert sha(p)==immutable[str(p)]
    print('actual_solver_exit=1 independent_area_audit=PASS retained_states='+str(len(rows))+' triangle_evaluations='+str(checks))
    print('saved_minimum_area_m2='+overall['minimum_serialized_area_m2']+' triangle='+str(overall['triangle_zero_based'])+' nodes='+str(overall['nodes_zero_based']))
    print('saved_minimum_path='+overall['path'])
    print('possible_unserialized_witness_area_m2='+str(overall['possible_unserialized_witness_area_m2']))
    print('saved_states_with_robust_failure='+str(len(failing))+' sole_definite_logged_numeric_failure=min_triangle_area')
    print('rounding_indeterminate_gates='+','.join(result['unresolved_predicate_observability']))

if __name__=='__main__':main()
