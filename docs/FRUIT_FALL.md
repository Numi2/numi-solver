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
changed source. A new 480-frame, two-replay run with valid initial packing is in progress;
it must be inspected before replacing historical evidence or claiming the
complete spill is fixed.

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
invalid-initialization diagnostic baseline. Fresh four-second CPU and Metal
pickup/settling runs use the corrected geometry and bind source and binary
hashes; their completed outcomes remain required.

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
