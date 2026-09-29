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
changed source. A new 480-frame, two-replay run is in progress; it must be
inspected before replacing the historical evidence or claiming the complete
spill is fixed.

The corrected two-second CPU FP64 pickup has completed two exact replays at
48 substeps and 32 iterations (`state_hash=0x513295fd7b0ff042`). Its release
mask is `3073`: fruits 0, 10, and 11 all end at their contact radii. Published
fruit/yarn penetration, ground penetration, and final strain violation are
zero; maximum published nonlocal yarn overlap is `0.746 um`. This is an
independent FP64 outcome, separate from the still-running Metal qualification.

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
