#!/usr/bin/env python3
"""Independent, read-only checks of a frozen area admission primitive.

Writes only inside this ignored review directory.  The original frozen source
and its qualification receipt are inputs, never edited by this program.
"""
from fractions import Fraction as F
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import sys

HERE = Path(__file__).resolve().parent
FROZEN = HERE / 'historical'
ORIGINAL_HASH = '5b9de1a9879657506a0a6b653cb05e13f7313e8f4ba44acdf276f2482cb09a1a'
Q = lambda x: F.from_float(float(x))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def sub(a, b):
    return tuple(x-y for x, y in zip(a, b))


def dot(a, b):
    return sum((x*y for x, y in zip(a, b)), F(0))


def direct_polynomial_value(m, t, gate):
    x = []
    for start, end in zip(m.start, m.end):
        x.append(tuple(Q(a) + t*(Q(b)-Q(a)) for a, b in zip(start, end)))
    c = cross(sub(x[1], x[0]), sub(x[2], x[0]))
    return dot(c, c) - 4*Q(gate)**2


def kinetic(v, masses):
    return sum((Q(m)*dot(tuple(Q(z) for z in row), tuple(Q(z) for z in row))/2
                for m, row in zip(masses, v)), F(0))


def independent_passive_check(result, x, v, masses, pins):
    assert result['accepted'] and result['status'] == 'passive'
    w = result['velocities']
    delta = kinetic(w, masses) - kinetic(v, masses)
    assert delta <= 0 and delta == F(result['represented_kinetic_change_exact_J'])
    assert all(math.hypot(*row) < 30 for row in v + w)
    assert all(w[i] == v[i] for i in pins)
    impulse = tuple(tuple(Q(masses[i])*(Q(w[i][k])-Q(v[i][k])) for k in range(3)) for i in range(3))
    momentum = tuple(sum((impulse[i][k] for i in range(3)), F(0)) for k in range(3))
    torque_rows = tuple(cross(tuple(Q(z) for z in x[i]), impulse[i]) for i in range(3))
    angular = tuple(sum((torque_rows[i][k] for i in range(3)), F(0)) for k in range(3))
    anchored_p = tuple(F(s) for s in result['constraint_impulse_on_anchors_exact'])
    anchored_l = tuple(F(s) for s in result['constraint_angular_impulse_on_anchors_exact'])
    residual_p = tuple(momentum[k]+anchored_p[k] for k in range(3))
    residual_l = tuple(angular[k]+anchored_l[k] for k in range(3))
    assert tuple(map(str, residual_p)) == tuple(result['represented_linear_balance_residual_exact'])
    assert tuple(map(str, residual_l)) == tuple(result['represented_angular_balance_residual_exact'])
    assert all(abs(z) <= F(result['represented_linear_balance_tolerance_exact']) for z in residual_p)
    assert all(abs(z) <= F(result['represented_angular_balance_tolerance_exact']) for z in residual_l)
    return max(abs(float(z)) for z in residual_p), max(abs(float(z)) for z in residual_l)


def run():
    assert sha(FROZEN/'area_admission.py') == ORIGINAL_HASH
    frozen = load('independent_frozen_area', FROZEN/'area_admission.py')
    candidate = load('independent_candidate_area', HERE/'area_admission.py')
    candidate_src = (HERE/'area_admission.py').read_text()
    original_src = (FROZEN/'area_admission.py').read_text()
    assert candidate_src == original_src.replace(
        'if B<q(GATE)**2 or represented_area(x)<GATE:',
        'if B<=q(GATE)**2 or represented_area(x)<=GATE:')
    assert (HERE/'qualification.exit').read_text().strip() == '0'

    G = candidate.GATE
    face_equal = ((0.,0.,0.), (1.,0.,0.), (0.,2*G,0.))
    zero = ((0.,0.,0.),)*3
    frozen_equal = frozen.passive_velocity(face_equal, zero, (1.,1.,1.))
    fixed_equal = candidate.passive_velocity(face_equal, zero, (1.,1.,1.))
    assert frozen_equal['accepted'] and frozen.cast(frozen.Motion(face_equal,face_equal))['status'] == 'initial_area_violation'
    assert not fixed_equal['accepted'] and fixed_equal['status'] == 'invalid'
    assert candidate.represented_area(face_equal) == G
    face_above = ((0.,0.,0.), (1.,0.,0.), (0.,2*math.nextafter(G, math.inf),0.))
    assert candidate.passive_velocity(face_above,zero,(1.,1.,1.))['accepted']
    face_below = ((0.,0.,0.), (1.,0.,0.), (0.,2*math.nextafter(G,0.),0.))
    assert not candidate.passive_velocity(face_below,zero,(1.,1.,1.))['accepted']

    rng = random.Random(0x123C0DE)
    root_count = 0
    max_width = F(0)
    for _ in range(48):
        L = rng.choice((F(1,4),F(1,2),F(1),F(2)))
        h0 = rng.choice((F(1,8),F(1,4),F(1,2),F(1)))
        h1 = rng.choice((F(1,8),F(1,4),F(1,2),F(1)))
        offset = tuple(F(rng.randrange(-8,9),8) for _ in range(3))
        start = (offset,(offset[0]+L,offset[1],offset[2]),(offset[0],offset[1]+h0,offset[2]))
        end = (start[0],start[1],(offset[0],offset[1]-h1,offset[2]))
        m = candidate.Motion(start,end)
        first = (h0 - 2*Q(G)/L)/(h0+h1)
        second = (h0 + 2*Q(G)/L)/(h0+h1)
        r = candidate.cast(m)
        assert r['status'] == 'first_boundary' and r['root_count'] == 2
        lo, hi = F(r['lower_time_exact']), F(r['upper_time_exact'])
        assert lo < first <= hi and hi < second
        assert hi-lo <= candidate.TIME_WIDTH and r['bracket_roots'] == 1
        assert direct_polynomial_value(m,first,G) == 0
        assert direct_polynomial_value(m,second,G) == 0
        assert direct_polynomial_value(m,lo,G) > 0
        assert direct_polynomial_value(m,hi,G) <= 0
        assert not r['accepted'] and r['total_proof_nodes'] <= candidate.CAP
        max_width = max(max_width,hi-lo)
        root_count += 1

    unit = ((0.,0.,0.),(1.,0.,0.),(0.,1.,0.))
    terminal = candidate.Motion(unit,((0.,0.,0.),(1.,0.,0.),(0.,2*G,0.)))
    tr = candidate.cast(terminal)
    assert tr['status'] == 'first_boundary' and tr['root_count'] == 1
    assert F(tr['lower_time_exact']) < 1 == F(tr['upper_time_exact'])
    tangent = candidate.Motion(((0.,0.,0.),(1.,0.,0.),(0.,-.5,2*G)),
                               ((0.,0.,0.),(1.,0.,0.),(0.,.5,2*G)))
    tang = candidate.cast(tangent)
    assert tang['status'] == 'first_boundary' and tang['root_count'] == 1
    assert F(tang['lower_time_exact']) < F(1,2) <= F(tang['upper_time_exact'])
    assert direct_polynomial_value(tangent,F(1,2),G) == 0
    assert direct_polynomial_value(tangent,F(1,2)-F(1,100),G) > 0
    assert direct_polynomial_value(tangent,F(1,2)+F(1,100),G) > 0
    quartic = candidate.Motion(unit,((0.,0.,0.),(-1.,0.,0.),(0.,-3.,0.)))
    four = candidate.cast(quartic)
    signs = [direct_polynomial_value(quartic,F(k,8),G) for k in (0,2,3,4,8)]
    assert [z>0 for z in signs] == [True,False,True,False,True]
    assert four['status'] == 'first_boundary' and four['root_count'] == 4
    assert F(four['upper_time_exact']) < F(1,4)

    # A separate component bound keeps c_z>=0.60 for every t; all these 3-D
    # affine faces have area>=0.30 m², independently of the candidate Sturm code.
    clear_count = 0
    for _ in range(64):
        end = tuple(tuple(start[k] + rng.randrange(-100,101)/1000 for k in range(3)) for start in unit)
        motion = candidate.Motion(unit,end)
        r = candidate.preventive(motion,(1.,2.,3.))
        assert r['accepted'] and r['status'] == 'clear'
        assert r['original_cast']['strict_polynomial_throughout']
        clear_count += 1

    admitted = 0
    max_p = max_l = 0.
    for _ in range(160):
        x = tuple(tuple(rng.randrange(-100,101)/100 for _ in range(3)) for _ in range(3))
        v = tuple(tuple(rng.randrange(-2900,2901)/100 for _ in range(3)) for _ in range(3))
        m = tuple(rng.choice((.5,1.,2.)) for _ in range(3))
        pins = (0,) if rng.randrange(4)==0 else ()
        if pins:v = (zero[0],v[1],v[2])
        r = candidate.passive_velocity(x,v,m,pins)
        if r['accepted']:
            p,l = independent_passive_check(r,x,v,m,pins)
            max_p,max_l = max(max_p,p),max(max_l,l)
            admitted += 1
        else:
            assert r['status'] in ('invalid','represented_passivity_failure','represented_reciprocity_failure','scene_speed_failure')

    # Fail-closed controls include local passivity with later global area loss,
    # moving supports, and an exhausted shared proof budget.
    future = candidate.preventive(candidate.Motion(unit,((0.,0.,0.),(1.,0.,0.),(0.,-1.,0.))),(1.,2.,3.))
    assert not future['accepted'] and future['response']['accepted']
    assert future['post_response_cast']['status'] == 'first_boundary'
    moving = candidate.preventive(candidate.Motion(unit,((.1,0.,0.),unit[1],unit[2])),(1.,1.,1.),(0,))
    assert not moving['accepted'] and moving['status'] == 'unsupported_moving_pin'
    exhausted = candidate.cast(candidate.Motion(unit,unit),candidate.Budget(candidate.CAP))
    assert not exhausted['accepted'] and exhausted['total_proof_nodes'] == candidate.CAP
    mutable = [[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]
    private = candidate.Motion(mutable,mutable)
    mutable[2][1] = 0.
    assert private.start[2][1] == 1. and candidate.cast(private)['accepted']

    result = {
      'status':'PASS_WITH_ONE_FROZEN_HELPER_DEFECT_CORRECTED_LOCALLY',
      'scope':'CPU affine-triangle path/velocity primitive only; no full cloth authority or actual drop repair',
      'frozen_source_sha256_before_after':ORIGINAL_HASH,
      'candidate_source_sha256':sha(HERE/'area_admission.py'),
      'candidate_qualification_source_sha256':sha(HERE/'qualify.py'),
      'candidate_qualification_actual_exit':0,
      'frozen_equal_gate_accepted':frozen_equal['accepted'],
      'candidate_equal_gate_accepted':fixed_equal['accepted'],
      'candidate_above_gate_accepted':True,
      'candidate_below_gate_accepted':False,
      'analytic_first_root_controls':root_count,
      'maximum_exact_root_bracket_width':str(max_width),
      'endpoint_root_counted':True,
      'double_root_counted':True,
      'quartic_four_roots_counted':True,
      'independently_bounded_3d_clear_controls':clear_count,
      'independent_represented_passive_checks':admitted,
      'maximum_represented_balance_residual_p':max_p,
      'maximum_represented_balance_residual_l':max_l,
      'future_crossing_rejected':True,
      'moving_pin_rejected':True,
      'exhausted_budget_rejected':True,
      'private_immutable_input_passed':True,
      'integration_blockers':['no full-scene common earliest event or incident-constraint authority',
        'no proof for native integrator rounding/history or actual saved owner velocities',
        'no all-affected yarn/bend/contact and external-work closure',
        'corrected CPU96 full drop remains FAIL']
    }
    assert sha(FROZEN/'area_admission.py') == ORIGINAL_HASH
    (HERE/'independent-result.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    return result


if __name__ == '__main__':
    print(json.dumps(run(),sort_keys=True))
