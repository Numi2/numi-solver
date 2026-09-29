# Moving yarn against an elastic surface: CPU geometry candidate

The next coupled example needs continuous contact between both yarn endpoints
and all three owning vertices of an elastic boundary face. The reviewed CPU
candidate advances that geometry beyond the isolated normal impulse:
**22 contact/clear/support fixtures**, **12 independent analytic temporal
checks**, and **two exact replays** pass. Five independently moving owners
resolve within 29 outer nodes. Maximum conservative spatial earliness is
**1.604 µm**, within the unchanged **2 µm** budget and **128-node** cap.

[Published source and qualification](assets/elastic-yarn-moving-ccd/published-evidence.json) ·
[Independent review](assets/elastic-yarn-moving-ccd/independent-review-evidence.json) ·
[CPU source](assets/elastic-yarn-moving-ccd/prototype.cpp).

This is a CPU geometry experiment. It is not integrated with native dispatch,
contact impulses, FEM stress advancement or a complete bag/elastic-fruit step.
Eight explicit rejection controls are retained. General incoming initial touch
and later grazing remain unresolved in this interval candidate.

## What an accepted interval proves

Each owner moves linearly between represented FP32 endpoints. Outward-rounded
FP64 projection intervals cover every yarn and triangle vertex throughout a
time interval. A positive separating-plane bound certifies its entire extent;
feature directions are candidates and must pass this complete support check.
Feasible material coordinates provide an upper distance bound at sampled times.

Search proceeds earliest first. A contact bracket has a proven contact at its
upper endpoint, a defined lower-endpoint normal, and both spatial earliness
and lower-endpoint upper gap within 2 µm. Every accepted lower endpoint also
has an instantaneous global lower gap at least −2 µm. Search requires a
strictly certified initial separation. An approximate feasible-pair distance
cannot establish that separation by itself. A complete-motion support bound
handles tested tangent/outward initial support. Degeneracy, invalid input,
arithmetic resolution and exhausted budgets produce explicit rejection.

The independent fixture oracle solves a box-times-simplex quadratic program
with FP64 active sets. Its sampled temporal search is a check of the named
fixtures; the interval support bounds supply the clearance certificate.
Accuracy at unbounded scales or translations is not established.

## Defects retained and repaired

Independent review found three concrete defects in the first prototype:

- A near-vertex barycentric shrink loop needed roughly 4.303 × 10¹⁸ predecessor
  steps. A single conservative complement bound replaces it. The actual
  near-vertex sweep completes in approximately 58 µs.
- A nonfinite plane could produce a positive clearance certificate because
  min/max hid NaN arithmetic. Finite/domain checks now reject it. No finite-owner
  end-to-end trigger is claimed for that direct-helper defect.
- At an extreme scale, a feasible-pair upper gap of +0.320 µm hid **557.184 µm
  of initial penetration**. The old code returned a contact bracket. The new
  global initial certificate rejects that exact state as unresolved.

The repair passes **56 helper controls**, **162 independent plane controls**,
**nine status/budget controls**, and **81 exact binary-rational checks** of
actual compiled simplex outputs. The original sources and failing logs remain
retained. Dedicated grazing and endpoint-only negative runs have actual exit 1;
those are rejected cases, not accepted collisions.

The later grazing frontier reaches a certified clear prefix of
0.24999999981810106, before exact first touch at 0.25, then returns unresolved.
A separate exact closed-support/root certificate is being qualified. It must
preserve feasible material membership and distinguish grazing from a closing
impact before it can support a runtime contact policy.

## Reproduce the published CPU candidate

From the repository root, compile without fast-math and with contraction off:

```sh
clang++ -std=c++23 -O2 -Wall -Wextra -Wpedantic -fno-fast-math -ffp-contract=off \
  -Iinclude docs/assets/elastic-yarn-moving-ccd/prototype.cpp \
  -o build/elastic-yarn-moving-ccd-probe
build/elastic-yarn-moving-ccd-probe
```

The same command with `bounded-controls.cpp` or `independent-controls.cpp`
compiles the additional controls. Public source copies rebuild byte-identically
to all three already executed owner binaries. The receipt binds both production
headers, compiler flags, source, qualified binary hashes and actual exits.

## Coupled step still required

A common transactional FEM/yarn step must evaluate both bodies' free motion,
apply reciprocal response to all five owners, reevaluate candidate FEM stress
and second-kick forces, then validate and publish both owning states and their
ledgers together. The [normal-contact module](ELASTIC_YARN_CONTACT.md) supplies
only the tested instantaneous velocity response. Position work, support/grip
reactions, frictional torque, material calibration, temporal/spatial convergence
and native complete-scene execution remain separate gates.
