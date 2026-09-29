# Whole-yarn static geometry and passive velocity response

The stronger shared FP32 sweep resolves all **1,299 seeded CPU sweeps**.
Its static distance and closing-velocity checks remain passing. The newest
[sweep receipt](assets/whole-yarn-sweep-refined-evidence.json) and
[host log](assets/whole-yarn-sweep-refined-host.log) bind the updated source.
The [previous receipt](assets/whole-yarn-static-math-evidence.json) remains
historical evidence of the earlier explicit unresolved results. This is **CPU
qualification**, with Metal compilation only; native execution and the complete
scene remain open.

## Whole axial segment

For endpoints `a,b`, radius `r` and `x(u)=(1-u)a+ub`, the geometry helper
minimizes the signed box/floor gap over the complete axial interval. Clear
endpoints alone cannot certify a clear yarn. Outward-rounded supporting-plane
lower bounds now certify the complete four-endpoint time/axial hull. A local
feature-plane candidate avoids rounded closest-point cancellation. Analytic
time proposals must pass the whole interval certificate; bounded scalar time
splits repair inconclusive proposals. A contact requires a complete signed
lower bound above -2 µm and a witness upper gap below +2 µm. Inconclusive
certificates and exhausted budgets remain explicit failures.

| CPU check | Measured result |
| --- | --- |
| Signed-gap samples / clearance certificates | 12,108 each; zero failures |
| Maximum signed-gap error / witness suboptimality | 0.162 / 0.144 µm |
| Primary sweeps | 781: 404 clear, 377 contacts, zero unresolved |
| General deforming sweeps | 518: 99 clear, 419 contacts, zero observed misses or unresolved results |
| Maximum positive arrival gap in primary set | 0.286 µm |
| Analytic impact position error | Maximum 0.113 µm in the analytic cases |
| Whole-interval closing-velocity queries | 1,008; zero failures; 26 solid-interior queries rejected |

The general sweeps use independently moving endpoints and an independent FP64
interval oracle. The former 25 primary and 129 general unresolved results are
retained in the comparison. Additional controls cover grazing, tangent/outward
motion, deforming endpoints, small radii, custom tolerances and extreme motion.
Four 1,024 m yarns remain explicitly unresolved against the unchanged 2 µm
precision budget. These finite tested sets do not establish unbounded-scale
accuracy or full-scene continuous-path safety. The primary arrival and analytic
errors above are separate measurements, not universal error bounds.

Signed plane division matters: positive projection uses an upper normal norm;
negative projection uses a lower norm. The former unsigned -radius fallback
could report clear after deep solid crossing when radius was below tolerance.
Independent controls retain that failure and check both axial orientations.
Deep overlap tolerated by a caller's custom tolerance may be certified clear
only if the whole path stays within that tolerance; it cannot masquerade as a
2 µm arrival contact.

The 128 outer geometry-iteration cap is unchanged. Each iteration may also
perform up to 32 scalar splits for each of two box planes and the floor. This
adds measurable work. The final short CPU benchmark on the same 1,299 queries
was **5.31 times slower** than the earlier primitive, excluding the independent
oracle. It includes the signed repair and does not qualify native throughput.


The velocity witness searches the complete supported axial interval for the
most closing normal velocity. Rounded-feature comparisons against dense FP64
sampling have maximum observed suboptimality 0.02093 mm/s. This is a sampled
comparison, not a proof of every manifold or feature transition. Four floor
regressions retain the former rounded nominal-root failure and now bracket the
actual rounded predicate in both axial orientations. No contact budget changed.

## Velocity slack and represented kinetic energy

`numiStaticResolveVelocityPair` evaluates each actual trial velocity
`v_i + inverseMass_i * J * gradient_i`, then applies that body's unilateral
point support. Outward normal velocity is available slack until the trial
becomes incoming. The scalar contact derivative brackets the impulse; a finite
friction cap can stop the solve before the derivative reaches zero. Stationary
pinned endpoints are supported. Moving kinematic endpoints reject because
the helper has no external-work ledger.

An independent outward-rounded upper bound checks the kinetic-energy change
of the represented input/output velocities. A positive or inconclusive bound
rejects the candidate. This prevents the old prematurely projected response
from creating energy while an endpoint still has outward velocity slack.

| Actual-mass kinetic-energy regression | Old algebra | New helper |
| --- | ---: | ---: |
| Normal impulse | **+0.0162373 J** | **-0.0152470 J** |
| Friction impulse | **+0.0182365 J** | **-0.0157823 J** |

All 16 physical cases, 10 invalid controls and two exact named-field replays
pass. Both old-algebra controls fail the independent energy oracle. Maximum
velocity error is 1.907 micrometres/s; maximum impulse, support-reaction and
mass-momentum closure errors are 19.87, 23.84 and 23.89 nNs. Maximum observed
kinetic-energy gain is zero. The latest CMake host checks pass, and the velocity result remains
byte-identical. The original velocity and combined receipts remain historical
bindings. The [current component receipt](assets/deformable-contact-host-components-evidence.json)
binds the new header, CPU binaries and tests. Separate Metal compilations
succeed; execution was not performed.

The pair helper uses point support at each body's position. It does not
assemble the full yarn/static manifold, a simultaneous normal/friction solve,
or the whole-scene reaction/work ledger. Native integration, the full loaded
bag replay, material calibration and cloth/elastic-fruit coupling remain open.

```sh
cmake --build build --target numi-solver-static-segment numi-solver-static-velocity
ctest --test-dir build -R '^static_contact\.(whole_yarn_host|velocity_slack_host)$' --output-on-failure
```
