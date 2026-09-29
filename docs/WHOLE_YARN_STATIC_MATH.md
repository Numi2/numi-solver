# Whole-yarn static geometry and passive velocity response

The shared FP32 helpers pass their completed **CPU** geometry and velocity
checks. Both CMake host tests pass, and their fresh output matches the retained
qualification logs byte for byte. This receipt contains **no native Metal
execution or full-scene qualification**. [Source and binary receipt](assets/whole-yarn-static-math-evidence.json),
[geometry log](assets/whole-yarn-static-geometry-host.log),
[velocity log](assets/whole-yarn-static-velocity-host.log),
[CMake test log](assets/whole-yarn-static-math-ctest.log).

## Whole axial segment

For endpoints `a,b`, radius `r` and `x(u)=(1-u)a+ub`, the geometry helper
minimizes the signed box/floor gap over `0 <= u <= 1`. It partitions the box
query at coordinate-boundary crossings, includes interior minima and retains
the contact parameter. Clear endpoints alone cannot certify a clear yarn.
An outward-rounded supporting-plane lower bound supplies the clearance
certificate. Sweeps bound segment motion by the larger endpoint displacement;
inconclusive bounds or exhausted iterations return an explicit failure status.
They must not be accepted as collision-free motion.

| CPU check | Measured result |
| --- | --- |
| Signed-gap samples / clearance certificates | 12,108 each; zero failures |
| Maximum signed-gap error / witness suboptimality | 0.162 / 0.144 micrometres |
| Primary sweep set | 781: 404 certified clear, 352 conservative contacts, **25 unresolved** |
| General deforming sweeps | 518; zero observed misses, FP64 oracle unresolved zero, **FP32 unresolved 129** |
| Maximum positive arrival gap in the primary sweep set | **2.976 micrometres** |
| Analytic impact position error | Maximum 0.160 micrometres in the analytic cases |
| Whole-interval closing-velocity queries | 1,004; zero failures; 26 solid-interior queries rejected |

The general sweeps use independently moving endpoints and an independent FP64
interval oracle. Safe unresolved classification passes the host test; it does
not establish a complete sweep solver. Conservative contacts can arrive at a
positive gap greater than the requested 0.1-micrometre contact tolerance.
The measured positive-gap maximum is from the primary sweep set. That
early-arrival precision and the 129 unresolved general sweeps remain open.
The analytic-case error is a separate measurement, not a bound for all sweeps.

The velocity witness searches the entire supported axial interval for the most
closing normal velocity. This avoids a flat-distance tie selecting an outward
endpoint while another supported portion closes. Rounded-feature comparisons
use dense independent FP64 sampling; maximum observed velocity suboptimality
is 0.02093 mm/s. This sampled comparison is not a proof of every possible
contact manifold or feature transition.

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
kinetic-energy gain is zero. A separate Metal compilation succeeds; execution
was not performed.

The pair helper uses point support at each body's position. It does not
assemble the full yarn/static manifold, a simultaneous normal/friction solve,
or the whole-scene reaction/work ledger. Native integration, the full loaded
bag replay, material calibration and cloth/elastic-fruit coupling remain open.

```sh
cmake --build build --target numi-solver-static-segment numi-solver-static-velocity
ctest --test-dir build -R '^static_contact\.(whole_yarn_host|velocity_slack_host)$' --output-on-failure
```
