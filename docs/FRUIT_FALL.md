# Fruit free flight and bench support

The September 29 correction preserves force-integrated velocity during XPBD
publication. Previously both cloth and fruit replaced velocity with
`(accepted_position - previous_position) / dt`. FP32 endpoint rounding then
fed back into every substep. Exact replay reproduced the error without
detecting it.

The new publication is `integrated_velocity +
(accepted_position - rounded_free_prediction) / dt`. The free prediction uses
the same fused operation as position advance. Unconstrained motion retains
gravity and air loads, while constraints still supply their actual corrections.
The FP64 oracle and separate CPU bag use the same decomposition.

For the authored inelastic support plane, an impact from outside ending
exactly at the radius publishes zero normal velocity. Fruit support impulse
includes the corresponding velocity change. This removes fractional-impact
descent and quantized support chatter without changing tangential contact or
prescribing any fruit trajectory. Initially penetrating probe states keep
their separate positional recovery behavior.

Reproduce the isolated native Metal check:

```sh
./build/numi-solver-cloth-metal --fruit-flight-probe
ctest --test-dir build -R 'cloth_bag.metal_(fruit_flight|internal_contact)' --output-on-failure -j1
```

The check advances the production kernels for 0.1 seconds at heights 1 and
8 metres, at the bag's `1/5760 s` timestep and its half-step refinement. With
no air or contact, it compares fruit and cloth velocity to gravity and position
to the discrete symplectic solution, then requires a bit-identical second
replay. Separate 0.3125-second vertical drops require exact plane support,
zero speed, and a normal impulse equal to weight times timestep.

| Observation | Before correction | After correction |
| --- | ---: | ---: |
| 1 m, refined: downward velocity after 0.1 s | 0.791016 m/s | 0.981007 m/s |
| 8 m, refined: downward velocity after 0.1 s | 0 m/s | 0.981007 m/s |
| Maximum velocity error over four free-flight cases | 0.981003 m/s | 0.000007034 m/s |
| Refined drop: final speed | 0.000018067 m/s after velocity-only correction | 0 m/s |
| Refined drop: support impulse error | 0.000007234 Ns after velocity-only correction | 0 Ns |

The analytic downward velocity is `0.981 m/s`. The corrected checks pass on
Apple M4 and M4 Pro, with no failure flags and exact replay. Maximum free-flight
position error is 0.224 mm over this short, translated FP32 check; preserving
velocity does not remove finite-precision absolute position storage.

Full-topology pickup is a separate qualification. The previously committed
GIF and trajectory numbers predate this correction. They do not qualify the
changed source. A new 480-frame, two-replay Metal run with valid initial
packing is in progress; it must be inspected before replacing historical
Metal evidence or claiming the complete native spill is fixed. The corrected
four-second CPU FP64 reference has separately passed, as recorded below.

The corrected two-second CPU FP64 pickup has completed two exact replays at
48 substeps and 32 iterations (`state_hash=0x513295fd7b0ff042`). Its release
mask is `3073`: fruits 0, 10, and 11 all end at their contact radii. Published
fruit/yarn penetration, ground penetration, and final strain violation are
zero; maximum published nonlocal yarn overlap is `0.746 um`. This is an
independent FP64 outcome from the original packing. The subsequent initial
geometry audit found invalid overlaps in that seed, so this result remains
a gravity/landing diagnostic and does not qualify the corrected starting scene.

Each exported Metal trajectory now also writes `PREFIX-fruits.csv`, with every
frame's fruit center, radius, linear/angular velocity, last-substep ground
impulse, and latched release bit. OBJ fruit comments include linear velocity.
These expose descent and landing even between the sparse rendered frames.
The renderer projects the z=0 support plane and individual fruit shadows from
the same world coordinates. Its displayed grid is presentation geometry;
the collision model remains an unbounded inelastic plane, without bench edges.

Audit the full native fruit trace without hiding a released fruit behind an
aggregate landing count:

```sh
python3 tools/audit_fruit_trace.py build/fruit-pickup-fruits.csv --expected-frames 480
```

The report includes the SHA-256 of the exact bytes parsed, so a concurrently
updated trace cannot silently change the evidence behind a saved report.
It reports each fruit's release, first observed support after release, maximum
downward speed, and final clearance and velocity. A partial trace returns
status 2, including when a concurrent copy ends inside a frame. A complete
trace report does not replace collision, force-balance, or replay qualification.

A latched mouth event does not establish that a fruit stays outside the moving
bag. Inspect the exported bag and fruit geometry separately:

```sh
python3 tools/audit_fruit_containment.py build/fruit-pickup-*.obj \
  --output build/fruit-pickup-containment.json
```

This report validates the exact authored render topology and closes its open
mouth with a virtual, oriented triangle fan. The solid-angle winding number
then reports each fruit center as inside, outside, or ambiguous. The virtual
cap is diagnostic geometry, not an added collision surface. It does not prove
full-sphere clearance, an exit or re-entry between snapshots, or physical
material calibration. Interpret it alongside the release history and contact
audit; a center inside the mesh and touching yarn must not be described as
unimpeded free flight merely because its release bit is latched.

For a fixed camera that includes every exported fruit throughout the replay,
write the complete ordered set of OBJ paths to a text file (relative paths are
resolved beside that file), then use the same framing list for every image:

```sh
swift tools/render_cloth_obj.swift build/fruit-pickup-480.obj build/fruit-pickup-480.png \
  trajectory --framing-list build/fruit-pickup-framing.txt
```

The camera fits the entire state set once rather than following or zooming with
the bag. Rendering a state outside that fixed framing is rejected. This fixes
the earlier `pickup-wide` camera cropping a landed fruit in the corrected FP64
state; the new surface/shadows and camera do not change any solver state.

Material calibration, finite bench geometry and edge collisions, impact
restitution calibrated to fruit/surface materials, and complete corrected
spill outcome remain separate work. These focused checks establish gravity
and inelastic support, not all scene fidelity.

## Local thickness at folded yarns

The retained failed refinement identified a gap in the two-hop self-contact
exclusion: nodes 861 and 957 have no direct yarn edge, but their centers came
within 0.887 mm while each endpoint owns a 4 mm cloth radius. The older
nonlocal capsule audit excluded that pair. The new independent node audit
reproduces its 7.113 mm diameter overlap and returns FAIL on the retained peak.

CPU and Metal now project all 5,754 unique two-hop node pairs without a direct
yarn edge to their authored 8 mm diameter. A relative swept-point contact
prevents an endpoint crossing from disappearing between discrete positions.
Inverse masses distribute free corrections; floor support removes blocked
vertical motion and transfers separation through available motion. Metal uses
12 disjoint-node color batches. Its FP64 oracle uses the same batch ordering.
The GPU config advances to ABI 14 to bind the new table counts explicitly.

```sh
./build/numi-solver-cloth-bag --local-node-probe
./build/numi-solver-cloth-metal --local-node-probe
python3 tools/audit_cloth_snapshot.py build/fruit-pickup-*.obj \
  --include-local-node-contacts --output build/fruit-pickup-contacts.json
```

The CPU probe covers the actual folded coordinates, unequal masses, floor
support and arrival, coincident endpoints, and a swept crossing. Six native
cases additionally check a fixed endpoint, physical separation, free center
of mass, exact replay, and FP64 response. Seven malformed native cases reject
without changing accepted particle bytes. Cold geometry remains overlap free.
The initial geometry, fruit free-flight/drop, and internal contact/oracle
checks also pass on Apple M4. See the [source/binary receipt](assets/cloth-local-node-evidence.json),
[CPU probe](assets/cloth-local-node-cpu-probe.log),
[Metal probe](assets/cloth-local-node-metal-probe.log), and
[independent negative audit](assets/cloth-local-node-negative-audit.json).

Both solver paths measure the new residual in published states. Native
grounded, spin, pickup and recorded trajectory qualifications require the
maximum over every frame of both replays to remain within 2 micrometres.
The repaired four-second CPU96 run is in progress. Full-scene qualification
and the unchanged 0.80 rad knot gate remain required; these focused checks
do not establish mesh-resolution convergence, calibrated local bending or
friction, yarn-interior contact, or complete energy closure.

The older ABI 13 native run has completed its first four-second replay. Only
fruit 8 has a latched release bit. Its center finishes inside the capped render
mesh, 0.928 m above its support radius, with no observed support after release.
This is not a qualifying spill or landing. Its second replay remains in
progress and must supply the actual terminal result.

## Starting contact geometry

The original cold seed had 5.12519 mm of sphere/yarn overlap and 4.48591 mm
of nonlocal yarn overlap before gravity or the grip moved. The worst yarn
pair connected particles 963:964 and 819:820 at the transition into the cuff.
Those positional corrections could inject velocity on the first step.

The body sag now fades smoothly into the cuff and has half its former
amplitude. Fruit centers 2, 4, 8, and 9 move by 6.0, 18.1, 2.0, and 6.3 mm
respectively, leaving at least 0.5 mm sphere/yarn and sphere/sphere clearance.
CPU and Metal author the same geometry and derive rest constraints from it.
Yarn thickness, fruit radii/masses, material parameters, solver iterations,
and grip motion are unchanged. This is collision-free authored packing;
it is not a measured fruit/bag specimen.

```sh
./build/numi-solver-cloth-metal --initial-state-probe
ctest --test-dir build -R 'cloth_bag.initial_geometry' --output-on-failure
```

The probe inspects the host-authored FP32 tables and all 4,149,792 eligible
yarn pairs. It reports zero initial fruit/yarn, fruit/fruit, nonlocal yarn,
and ground overlap on M4 and M4 Pro. Normal Metal execution runs the same
check before creating its device and rejects an invalid seed. A contract-valid
6 mm yarn radius is correctly rejected because this particular seed intersects
fruit and the support plane at that thickness.

The valid cold seed has no contact on its first substep. The oracle comparison
now requires matching contact presence/absence, while the dedicated fruit/yarn
CCD and yarn/yarn CCD cases still require positive impulses and friction,
slip reduction, and the existing geometric error bounds. Geometry, native
free flight/drop, internal contact, and default material parity checks pass.

The earlier full Metal run was deliberately stopped and retained as an
invalid-initialization diagnostic baseline. The corrected four-second CPU
pickup/settling reference has completed two exact final-state replays and
passed its physical gates. The full Metal outcome remains required. Both
runs retain source and binary hashes; the CPU qualification log and frame
fingerprints are linked below.

## Independent exported-state contact audit

```sh
python3 tools/audit_cloth_snapshot.py build/fruit-pickup-*.obj \
  --yarn-radius 0.004 --output build/fruit-pickup-contacts.json
```

The standalone auditor validates all 2,880 render triangles before reconstructing
the exact 2,904 axial yarns and graph-derived two-hop self-contact exclusions.
It measures sphere/yarn, sphere/sphere, nonlocal capsule/capsule, and ground
overlap independently of the solver. Inflated segment AABBs in a spatial grid
produce a conservative self-contact candidate set; exact segment distance
decides overlap. Every supplied OBJ is bound to its SHA-256. Its tolerance
is 2 micrometres. A material using another yarn radius must supply that value.

It independently reproduces both original initial defects (5.12519 mm
fruit/yarn and 4.48591 mm nonlocal yarn), and confirms zero overlap in the
corrected cold seed. Corrected native frames 10 and 20 are within tolerance
(under 14 nm fruit/yarn overlap). These are sampled published states; the
auditor does not certify intervening substeps, strain, or dynamics. The full
trajectory and the native solver's failure/replay/strain gates remain required.

## Settling beyond the presentation and diagnostic bounds

The first corrected four-second FP64 run matched two exact replays
(`state_hash=0x23465c2d4a1627f4`). Fruits 0, 4, 10, and 11 all finished
with `center.z == radius` and zero vertical velocity. All 49 exported states
pass the independent contact audit; maximum sampled fruit/yarn, fruit/fruit,
and nonlocal yarn overlaps are below 8 nm, and sampled ground overlap is zero.
The native reference's published-state certificate has maximum sphere/yarn
overlap 0.200 um, nonlocal yarn overlap 0.960 um, and zero ground/strain
violation. These are rolling landings, not zero-total-speed rest.

That run still returned FAIL: the old numerical-escape classifier flagged
fruits 4 and 11 solely for travelling beyond five metres during the longer
settling interval (`escaped_mask=2064`). This was an arbitrary diagnostic
box, while the collision plane and free space are unbounded. The reference
now separately reports `outside_diagnostic_bounds_mask`. A finite fruit with
a classified mouth exit in pickup/recorded scenarios may travel outside that
box. Unreleased fruit retains the original containment bounds. Nonfinite state,
invalid radius, below-plane positions, and the existing independent contact,
strain, speed, orientation and force gates still reject invalid runs. Focused
checks retain those rejections and accept valid far-field rolling and flight.
A fresh source/binary-bound four-second replay has now passed. The physical
hash remains `0x23465c2d4a1627f4`; all 49 exported OBJ states and the final
OBJ match the earlier run byte for byte. The old FAIL is retained. The fresh
run reports `escaped_mask=0`, `outside_diagnostic_bounds_mask=2064`, and all
four released fruits at their contact radii with zero vertical velocity.

The completed trajectory also exposed a display mismatch: the renderer drew
a finite +/-4 m square over the unbounded collision plane. Landed fruit beyond
that square appeared to float above the background. The rasterizer now draws
the support plane across the complete orthographic viewport, inverts the world
projection to cover the visible grid, and retains the world-position shadows.
Grid density is bounded; with a shared trajectory framing list the grid and
camera stay fixed for every frame. It changes no simulated positions.

The corrected four-second FP64 GIF, video, log, and source/binary/frame
fingerprints are now published in `docs/assets/cloth-pickup-*`. They qualify
the CPU reference described above. The full Metal pickup and settling run
remains live and requires its own complete outcome and two-replay agreement.

## Full-scene half-timestep qualification remains failed

The September 29 four-second CPU run at 96 substeps has now completed two
exact final-state replays (`state_hash=0xdf5b1f22c35cf526`). It returns FAIL:
maximum knot-angle error is `0.818173682 rad`, above the unchanged strict
`0.80 rad` gate. The [failed log](assets/cloth-pickup-refined-failed.log) and
[source, binary and exported-state receipt](assets/cloth-pickup-refined-failed-evidence.json)
retain that result. This does not qualify full-scene temporal convergence.

All 49 exported states independently pass the 2 µm contact tolerance.
Published sphere/yarn overlap is `0.047 µm`, nonlocal yarn overlap
`0.432 µm`, and ground/strain violation is zero. Fruits 4, 9, 10, and 11 end
outside the virtually capped render mesh at their ground radii with zero
vertical velocity. Those landing and sampled-contact observations do not
erase the knot failure. The maximum knot peak requires further diagnosis;
the material coefficients and acceptance threshold remain unchanged.

`--knot-trace PATH` records every new maximum with its exact frame, constraint,
current/rest angles, and four endpoint indices. `--dump-knot-peak PATH`
retains the corresponding complete solver state, even when the peak lies
between the regular sparse snapshots. These are observation controls; the
20-frame observer run reproduces the earlier final and peak-frame OBJ bytes
and final state hash `0xfbe152b73f646f19` exactly. That short diagnostic does
not qualify a spill or replace the failed full run.

The full observer run reproduces the earlier failed run's final OBJ byte for
byte and retains `state_hash=0xdf5b1f22c35cf526`. Its exact peak is frame 162
(`1.35 s`), knot 861, with warp endpoints 861/957 and weft endpoints 908/910.
The native current/rest angles are `2.231100316 / 1.412926634 rad`.
[Trace](assets/cloth-knot-peak-trace.csv),
[complete peak geometry](assets/cloth-knot-peak.obj), and
[source-bound diagnostic receipt](assets/cloth-knot-peak-evidence.json) retain
the actual failing state.

The warp outer-endpoint chord shrinks from `20.236 mm` at rest to `0.887 mm`,
while its two arms remain `10.291 / 9.577 mm`. The outer endpoints are closer
than the authored `8 mm` yarn diameter. Their two-hop adjacency excludes them
from the current self-contact checks, so the existing nonlocal contact audit
still passes at this peak. This exposes a local thickness and tangent-conditioning
defect for the next mechanics repair. It is not evidence that the failed
knot gate has been repaired or that a physical yarn-curvature law is calibrated.
