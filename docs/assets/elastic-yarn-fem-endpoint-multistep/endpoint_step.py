#!/usr/bin/env python3
"""Selected-face endpoint contact extension of the frozen shared normal block.

The old block solved the two endpoint velocities but retained the centroid
witness admission from its predecessor. Once the actual closest point moves
to a yarn endpoint, that unrelated centroid can be >2 um from the true
minimum. This class uses the actual segment/triangle witness for admission;
the same two endpoint rows, 2x2 reciprocal LCP and all constitutive/body
operators remain frozen. No edge/vertex feature or CCD is implemented.
"""
import importlib.util
import math
import pathlib

P = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('frozen_normal_block', P / 'step.py')
N = importlib.util.module_from_spec(spec)
spec.loader.exec_module(N)
B = N.B
FROZEN_SHA = 'dcfaeca294146fbec3e161dce05a7583d0ffdf34e8e181e2a008c92a0dc22176'
assert B.sha(P / 'step.py') == FROZEN_SHA


class Model(N.Model):
    def __init__(self):
        super().__init__()
        self.inputs[str(pathlib.Path(__file__))] = B.sha(pathlib.Path(__file__))

    def geometry(self, x):
        own = [x[i] for i in self.row]
        a, b, *tri = own
        edges = [B.sub(tri[1], tri[0]), B.sub(tri[2], tri[0]), B.sub(tri[2], tri[1])]
        max_edge2 = max(B.dot(e, e) for e in edges)
        cross2 = B.dot(B.cross(edges[0], edges[1]), B.cross(edges[0], edges[1]))
        area_gate = 128 * B.EPS32 * B.EPS32 * max_edge2 * max_edge2
        if not B.finite(sum(own, [])) or max_edge2 <= 0 or cross2 <= area_gate:
            raise N.Reject('invalid-contact-area', {'cross_squared': cross2, 'gate': area_gate})

        # Both supported endpoint projections are computed from the same five
        # actual FP64 owners, and the frozen block uses exactly these rows.
        rows = self.endpoint_rows(x)
        oracle = B.closest(a, b, tri)
        if not oracle['valid'] or not B.finite([oracle['distance'], oracle['u']] + oracle['bary']):
            raise N.Reject('endpoint-global-oracle-unresolved', oracle)
        u, beta = oracle['u'], oracle['bary']
        if u < -64 * B.EPS or u > 1 + 64 * B.EPS:
            raise N.Reject('endpoint-global-u-infeasible', oracle)
        if min(beta) <= 64 * B.EPS or max(beta) >= 1 - 64 * B.EPS:
            raise N.Reject('unsupported-triangle-feature', oracle)
        yp = B.add(B.mul(a, 1 - u), B.mul(b, u))
        tp = B.sumv([B.mul(tri[k], beta[k]) for k in range(3)])
        delta = B.sub(yp, tp)
        distance = B.norm(delta)
        cutoff = 8 * B.EPS32 * max(1., B.norm(yp), B.norm(tp))
        if distance <= cutoff:
            raise N.Reject('undefined-endpoint-global-normal', {'distance': distance, 'cutoff': cutoff})
        # The active-set witness and independently reconstructed actual-owner
        # witness must describe one geometry; keep the original 2 um budget.
        if abs(distance - oracle['distance']) > B.GAP_GATE:
            raise N.Reject('endpoint-global-witness-disagree', {'distance': distance, 'oracle': oracle})
        gap = distance - self.radius
        if gap < -B.GAP_GATE:
            raise N.Reject('contact-manifold/gap', {'distance': distance, 'gap': gap,
                                                     'radius': self.radius, 'oracle': oracle})
        if gap > B.GAP_GATE:
            raise N.Reject('unsupported-separated-contact', {'gap': gap, 'oracle': oracle})
        normal = B.mul(delta, 1 / distance)
        if any(B.dot(normal, row['normal']) < 1 - 512 * B.EPS for row in rows):
            raise N.Reject('unsupported-normal-orientation', {'oracle': oracle})
        return {'distance': distance, 'global_distance': oracle['distance'], 'gap': gap,
                'row_gap': gap, 'normal': normal, 'u': u, 'beta': beta[:],
                'oracle': oracle, 'triangle_area_m2': .5 * math.sqrt(cross2),
                'radius': self.radius, 'endpoint_rows': rows,
                'contact_witness_operator': 'actual-FP64-segment-triangle-closest-supported-endpoint'}


advance = N.advance
velocity_scan = N.velocity_scan
