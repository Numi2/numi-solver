# Numi Solver

Numi Solver is the independent Apple Metal home of the Temporal Cone contact
solver extracted from Numi Lab's `numisolver` branch.

This repository intentionally contains only the owning Metal solver source and
its pointer-free GPU ABI headers. It does not contain Numi Lab's robot models,
training runtime, tasks, applications, research artifacts, rendering runtime,
or website. Files under `docs/assets` are generated solver evidence, not an
application scene.

## Pick up the bag and spill the fruit

![A deterministic Metal cloth replay lifting a woven produce bag by its reinforced top opening seam and spilling fruit onto the ground](docs/assets/cloth-metal-pickup-spill.gif)

This 49-frame GIF is rasterized from one continuous four-second Apple Metal
solver trajectory. The first two seconds lift, snap, and recover the virtual
handle; the final two seconds hold it fixed while the released fruit fall and
reach the plane. The orange ring acts through finite compliance on ten
neighboring knots across the two-row reinforced top cuff. It is not attached to the bottom,
and no cloth knot or fruit is prescribed. All 1,465 cloth particles and twelve
fruits remain dynamic. The fixed wide camera changes no solver state and keeps
the seam, closed woven bottom, top opening, ground, and spilled fruit visible.

On an Apple M4 Pro, two complete 480-frame replays each advanced 48
device-resident substeps per frame. Both were failure-free and bit-identical at
all 480 published frame hashes and in the final physical buffers.
`released_mask=1537` records three full-sphere exits through the 48-knot top
mouth; two released fruits finish with `center.z == radius` on the plane.
Final strain violation, cloth/fruit ground penetration, and nonlocal yarn
self-penetration are all `0`. The first and second runs reported
`7086.040673292242 s` and `7085.412014581729 s` of GPU command time,
respectively, and the maximum observed resident set was `914576 KiB`.
This is executable Metal physics evidence, not a posed animation or a GPU
performance comparison.

| Grounded start | Top-seam lift | Mouth exit | Released descent | Two grounded |
|:--:|:--:|:--:|:--:|:--:|
| ![Grounded woven bag before the Metal seam lift](docs/assets/cloth-metal-pickup-0.png) | ![Metal cloth bag hanging from the highlighted top cuff](docs/assets/cloth-metal-pickup-60.png) | ![Fruit crossing the open 48-knot mouth in the Metal replay](docs/assets/cloth-metal-pickup-160.png) | ![Released fruit descending while the Metal handle becomes stationary](docs/assets/cloth-metal-pickup-240.png) | ![Two released fruit physically grounded after the fixed-handle settling tail](docs/assets/cloth-metal-pickup-480.png) |

### Independent FP64 reference

![A deterministic FP64 cloth reference lifting a produce bag from a compliant rim patch and spilling fruit onto the ground](docs/assets/cloth-pickup-spill.gif)

The separate two-second CPU FP64 reference records four robust mouth exits and
keeps all four spilled fruits on the plane. Its 48-substep replay passed with
`released_mask=3344`, `escaped_mask=0`, `0.206217584` maximum warp
extension, `0.089365346` maximum weft strain, `0.021853543` maximum
woven-bottom extension, `237.766842811 N` peak grip force, `0.002 um` final
primitive overlap, zero final strain-limit violation, and bit-identical state
hash `0xe6aa91e33438537d`. It is an independent equation-level and trajectory
reference; chaotic long-contact trajectories are not required to match the
Metal state bit for bit.

| Start | Seam lift | Loaded cuff | Mouth exit | Four remain out |
|:--:|:--:|:--:|:--:|:--:|
| ![Grounded bag before the FP64 seam lift](docs/assets/cloth-pickup-0.png) | ![FP64 bag hanging from the highlighted top cuff](docs/assets/cloth-pickup-60.png) | ![Fruit loading the deforming FP64 cuff before the downward snap](docs/assets/cloth-pickup-120.png) | ![Fruit crossing the open FP64 mouth](docs/assets/cloth-pickup-160.png) | ![Four FP64 released fruit contacting and rolling on the plane](docs/assets/cloth-pickup-200.png) |

## Grounded cloth produce-bag replay

![The complete Metal cloth produce bag after one second of free gravity, cloth, fruit, and plane contact](docs/assets/cloth-metal-grounded-120.png)

This frame is rasterized from the complete one-second Apple Metal grounded
trajectory. The seam grip is inactive: all 1,465 cloth particles and twelve
fruits move only through gravity, the woven constraints, contact, friction, and
air loads. Two 120-frame replays, each with 48 device-resident substeps per
frame, matched all frame hashes and final physical buffers exactly. The final
state had 42 cloth particles supported at `z=0.004000000190 m`, no released or
escaped fruit, `0` strain violation, `0` ground penetration, `1.08e-8 m`
maximum nonlocal yarn overlap, and `0.091619409 m` maximum sampled
internal-distance change. Its `0.336875938 J` final kinetic energy means this
is a grounded one-second trajectory, not a claim of perfect rest.

### Independent FP64 grounded reference

![A low cotton-net produce bag with a folded cuff, scalloped ground skirt, and twelve fruit spheres after one simulated second](docs/assets/cloth-produce-bag.png)

This PNG is rasterized directly from the actual deterministic FP64 solver state;
it is not concept art or a hand-posed scene. The replay advances a free 48 by 28
cloth wall, a structured 13 by 13 woven bottom, and twelve dynamic fruit
spheres for one second. The
AppKit rasterizer adds cotton-fiber strokes between control points but does not
move the exported state.

The qualified replay passed with no escaped or spilled fruit, no collapsed
render triangles, `0.004774618 m` maximum fruit/yarn contact correction,
`0.114696768` maximum warp extension, `0.001 um` maximum published
fruit/yarn penetration, `0.934 um` final nonlocal yarn overlap, zero final
strain-limit violation, and bit-identical replay hash
`0x58bbb2338d1a5369`. Every resolved tangent impulse remained inside its Coulomb
cone. This is CPU FP64 cloth evidence, not Metal-performance or Temporal Cone
cloth-contact evidence.

### Grab a compliant rim patch and spin

![A deterministic Apple Metal cloth bag deforming while its compliant top-opening seam patch follows a circular trajectory](docs/assets/cloth-metal-spin.gif)

The orange ring marks the target of the compliant ten-knot, two-row cuff
attachment. All cloth nodes and all twelve fruits remain dynamic. These five
fixed-camera frames come from one continuous 0.5-second Apple Metal trajectory;
they are not separately posed simulations.

| `t=0.000 s` | `t=0.125 s` | `t=0.250 s` |
|:--:|:--:|:--:|
| ![Initial airborne Metal cloth bag](docs/assets/cloth-metal-spin-0.png) | ![Metal cloth bag beginning to lag behind its moving seam grip](docs/assets/cloth-metal-spin-15.png) | ![Metal cloth bag folding around the compliant top-cuff patch](docs/assets/cloth-metal-spin-30.png) |

| `t=0.375 s` | `t=0.500 s` |
|:--:|:--:|
| ![Open Metal cloth bag twisting under circular seam motion](docs/assets/cloth-metal-spin-45.png) | ![Metal cloth bag lagging below the seam at the end of the orbit](docs/assets/cloth-metal-spin-60.png) |

Two complete 60-frame Metal replays matched every frame hash and the final
physical buffers exactly. No fruit released or escaped. Final strain violation
and nonlocal yarn overlap were both `0`; the seam traveled `0.400488745 m`
end-to-end with `0.011406752 m` maximum cloth lag. The cloth's maximum sampled
internal-distance change was `0.275368664 m`, which cannot come from rigid-body
motion. The first and second runs reported `766.298843375 s` and
`766.806226959 s` of GPU command time. This is executable Metal physics
evidence, not an interactive mouse-grab UI or a posed animation.

#### Independent FP64 spin reference

The separate 48-substep FP64 spin replay passed with `0.158858687` maximum
warp extension, `0.030219467` maximum weft strain, zero numerical escapes or
robust releases, `157.431556610 N` peak attachment force, maximum published
yarn overlap `0.990 um`, and bit-identical hash `0xc0eb9fe5dd84d638`.
Explicit air loads
reached `0.142724430 N` on a yarn segment and `0.027419964 N` on a fruit while
removing `0.734375455 J` of relative kinetic energy. The contents press into
the deforming bag during this half-second orbit; the separate pickup replay
contains the spill.

Reproduce the FP64 reference images from executable state:

```sh
./build/numi-solver-cloth-bag \
  --scenario grounded --steps 120 --substeps 24 --iterations 32 --replays 2 \
  --dump-obj build/cloth-produce-bag-1s.obj

swift tools/render_cloth_obj.swift \
  build/cloth-produce-bag-1s.obj docs/assets/cloth-produce-bag.png

./build/numi-solver-cloth-bag \
  --scenario spin --steps 60 --substeps 48 --iterations 32 --replays 2 \
  --dump-frames build/cloth-spin

for step in 0 15 30 45 60; do
  swift tools/render_cloth_obj.swift \
    "build/cloth-spin-${step}.obj" "docs/assets/cloth-spin-${step}.png"
done

./build/numi-solver-cloth-bag \
  --scenario pickup --steps 240 --substeps 48 --iterations 32 --replays 2 \
  --dump-frames build/cloth-pickup --dump-every 10

mkdir -p build/cloth-pickup-png
for step in $(seq 0 10 240); do
  swift tools/render_cloth_obj.swift \
    "build/cloth-pickup-${step}.obj" \
    "build/cloth-pickup-png/frame-${step}.png" pickup
done

swift tools/compose_cloth_gif.swift \
  docs/assets/cloth-pickup-spill.gif 0.08 \
  $(for step in $(seq 0 10 240); do
    printf '%s ' "build/cloth-pickup-png/frame-${step}.png"
  done)
```

The complete Metal grounded and circular-seam trajectories run two exact
replays and export replay one's states:

```sh
./build/numi-solver-cloth-metal \
  --replays 2 --iterations 32 --strain-sweeps 3 \
  --grounded-prefix build/metal-grounded --grounded-steps 120 \
  --grounded-dump-every 10

./build/numi-solver-cloth-metal \
  --replays 2 --iterations 32 --strain-sweeps 3 \
  --spin-prefix build/metal-spin --spin-steps 60 \
  --spin-dump-every 5

mkdir -p build/metal-spin-png
for step in $(seq 0 5 60); do
  swift tools/render_cloth_obj.swift \
    "build/metal-spin-${step}.obj" \
    "build/metal-spin-png/frame-${step}.png"
done

swift tools/compose_cloth_gif.swift \
  docs/assets/cloth-metal-spin.gif 0.08 \
  $(for step in $(seq 0 5 60); do
    printf '%s ' "build/metal-spin-png/frame-${step}.png"
  done)
```

The opt-in Metal trajectory uses the same 240-frame, 48-substep pickup motion,
holds the final handle for 240 more frames so spilled fruit can reach the
plane, executes two full replays, and exports every tenth frame from replay
one:

```sh
./build/numi-solver-cloth-metal \
  --replays 2 --iterations 32 --strain-sweeps 3 \
  --pickup-prefix build/metal-pickup --pickup-steps 480 \
  --pickup-dump-every 10

mkdir -p build/metal-pickup-png
for step in $(seq 0 10 480); do
  swift tools/render_cloth_obj.swift \
    "build/metal-pickup-${step}.obj" \
    "build/metal-pickup-png/frame-${step}.png" pickup-wide
done

swift tools/compose_cloth_gif.swift \
  docs/assets/cloth-metal-pickup-spill.gif 0.08 \
  $(for step in $(seq 0 10 480); do
    printf '%s ' "build/metal-pickup-png/frame-${step}.png"
  done)
```

## Solver boundary

The initial `src/metal/MetalWorldContact.metal` import was preserved
byte-for-byte from the source revision recorded in
[PROVENANCE.md](PROVENANCE.md). Independent development is tracked from that
snapshot. The file owns the complete contact pipeline required by Temporal
Cone, including deterministic island and tile construction, Wave8/16/32 cohort
selection, coupled normal/tangent cone updates, distributed-island reduction,
stiff-island ordered replay, warm-start publication, and transactional status
reduction.

The solver still speaks the original versioned MetalWorld ABI. Narrow,
pointer-free ABIs expose its local cone mathematics, packed contact islands,
response-column assembly, and rigid-body response/publication path. They call
the production Metal helpers; they do not carry a second shader
implementation.

## Build

Requirements:

- Apple Silicon macOS
- CMake 3.28 or newer
- Xcode/Command Line Tools with Metal 4 support

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

Release is also selected automatically when a single-configuration generator
is used without an explicit build type. The explicit-yarn FP64 cloth oracle is much
slower without compiler optimization; timing from an unoptimized build is not
qualification evidence.

The build produces `build/shaders/NumiTemporalCone.metallib` using `-O3` and
`-fno-fast-math`, plus eleven native harnesses:

- `build/numi-solver-math` for isolated local cone blocks;
- `build/numi-solver-islands` for dense-versus-streamed coupled
  1/2/4/8/16/32-contact islands;
- `build/numi-solver-assembly` for GPU response-column assembly followed by
  a streamed solve on one command buffer.
- `build/numi-solver-rigid` for rigid contact frames and mass/inertia through
  deterministic velocity publication on one command buffer.
- `build/numi-solver-braided-bag` for an end-to-end deformable braided bag
  containing six falling balls, with matched dense and streamed trajectories.
- `build/numi-solver-cloth-bag` for a deterministic FP64 explicit-yarn
  reference with a free folded rim, 2,904 axial yarn segments, 1,369 compliant
  crossing-angle knots, 2,834 three-knot bend chords, twelve-fruit sphere/yarn
  contact and friction, continuous yarn/yarn collision, unilateral extension
  limits, maximum-dissipation yarn friction, a stiff authored two-row cuff, and
  a compliant ten-knot top-seam grip patch. Cylinder/sphere air loads,
  load-dependent cloth/ground friction, and
  load-dependent fruit rolling resistance replace global velocity damping.
  Its 2,880 triangles are export/raster and area-certificate geometry, never
  collision or mechanical constraints. This is a CPU reference, not
  Metal-performance evidence.
- `build/numi-solver-cloth-metal` for the versioned device-owned gravity,
  graph-colored axial XPBD, crossing-angle knots, ground-aware yarn bending,
  unilateral extension limit, ten-knot seam grip, twelve-fruit translation,
  all 66 fruit-pair normal contacts, cloth/fruit ground projection, all 34,848
  present and swept fruit/yarn candidate geometries, deterministic
  ground-aware fruit/yarn normal response, all 4,149,792 graph-excluded
  yarn/yarn candidates, spatially compacted current and exhaustive swept
  yarn/yarn self-contact, fixed contact/strain reconciliation, and velocity
  publication on the full 1,465-particle/2,904-segment topology. It then applies
  cloth/ground, yarn/yarn, fruit/yarn, fruit/fruit, and fruit/ground Coulomb
  friction plus load-capped fruit rolling resistance and normalized fruit
  orientation integration. Full-yarn cylinder drag and fruit translational and
  rotational air decay execute before position advance. Ordered-mouth release
  classification distinguishes a physical spill through the top cuff from a
  numerical escape.
  It gates FP64 equation and impulse error, sphere/yarn and yarn/yarn tunneling
  response probes, sub-micrometer final self-overlap, and exact replay;
  the complete bag trajectory remains open.
- `build/numi-solver-articulated` for canonical articulated mass/Jacobian
  factorization, response-column solves, cone contact, and deterministic
  generalized-velocity publication on one command buffer.
- `build/numi-solver-articulated-capacity` for the complete 32-DoF,
  32-contact, 1,024-block articulated frontier and the complete admitted
  streamed inverse-ABA transaction.
- `build/numi-solver-articulated-conditioning` for deterministic rejection
  and rollback of forward-inaccurate ill-conditioned mass response.
- `build/numi-solver-articulated-zero-armature` for explicit rejection of an
  inverse response with no positive armature lower bound.

Run the default FP64 comparisons and deterministic replays:

```sh
./build/numi-solver-math
./build/numi-solver-islands
./build/numi-solver-assembly
./build/numi-solver-rigid
./build/numi-solver-braided-bag
./build/numi-solver-cloth-bag
./build/numi-solver-cloth-metal
./build/numi-solver-articulated
./build/numi-solver-articulated-capacity
./build/numi-solver-articulated-conditioning
./build/numi-solver-articulated-zero-armature
ctest --test-dir build --output-on-failure
```

The local harness accepts `--cases N --replays N --iterations N` and
`--isotropic`. The island, assembly, rigid, and articulated harnesses accept
`--islands N` and `--replays N`. The bag accepts `--environments N`,
`--steps N`, `--replays N`, `--timestep DT`, optional `--refinement`,
`--preset baseline|stiff-braid|low-cfm|high-friction`, and optional
`--dump-obj PATH`. `--require-speedup` optionally makes the within-run
streamed-over-dense timing comparison an exit gate; normal physics
qualification reports timing without conflating it with correctness. Together
they check separating, impact, sticking, sliding,
anisotropic friction, near-rank-deficient response, capped impulse, polar
boundary, exact one-axis friction, extreme scale, sparse topology, full block
capacity,
shared response, missing coupling, response asymmetry, rigid momentum and
kinetic-energy budgets, implicit contact-law regularization, thresholded
restitution, penetration recovery, contact-frame validity, and transactional
velocity rollback. The articulated gates additionally check analytic and
finite-difference serial-chain Jacobians, independent FP64 mass matrices,
factor-solved response columns, exact infinity-norm condition estimates,
generalized kinetic energy, 32-DoF/32-contact capacity, typed conditioning
rejection, O(n) inverse-ABA actions, bit-identical kinematics/contact-frame
preparation, inverse-response assembly and publication, and invalid
state/frame/material rollback.

The explicit-yarn cloth reference accepts `--steps N`, `--substeps N`,
`--iterations N`, `--replays 1|2`, `--timestep DT`,
`--scenario grounded|spin|pickup`, and `--dump-obj PATH`. Qualification requires
two replays. `--rolling-probe` executes the analytic solid-sphere slide-to-roll
oracle; `--rolling-resistance-probe` checks load-capped rolling torque without
damping vertical spin; `--aerodynamics-probe` checks analytic yarn/fruit drag,
energy removal, spatial subdivision invariance, exact isolated temporal
refinement, and co-moving air;
`--yarn-mechanics-probe` certifies swept sphere/yarn contact and a
finite-compliance knot-angle solve; `--self-ccd-probe` certifies continuous
yarn/yarn contact; and `--strain-probe` certifies the unilateral extension
ceiling. `--deformable-response-probe` checks true endpoint-mass transfer with
free and plane-supported yarn. `--self-friction-probe` independently checks
yarn/yarn sticking, sliding, momentum conservation, dissipation, cone
feasibility, and replay; `--cloth-ground-friction-probe` checks loaded sticking,
cone-limited sliding, and zero-load invariance. `--dump-frames PREFIX` exports five evenly spaced states
from the first authoritative trajectory; `--dump-every N` instead exports
every Nth step plus the final state. `grounded` settles an open produce bag on
a plane, `spin` begins airborne and drives a compliant ten-knot cuff patch
around a smooth orbit, and `pickup` lifts that same grip patch from the grounded state and
pours fruit onto the plane. Its FP64 mechanics and evidence boundary are
separate from the Metal harnesses.

The Metal cloth harness accepts `--replays N`, `--iterations N`,
`--strain-sweeps N`, and opt-in `--grounded-prefix`, `--grounded-steps`,
`--grounded-dump-every`, `--spin-prefix`, `--spin-steps`,
`--spin-dump-every`, `--pickup-prefix`, `--pickup-steps`, and
`--pickup-dump-every` trajectory arguments. It reconstructs the full cloth
internal-constraint/grip topology, validates conflict-free graph colors,
compares every published value
against an independent FP64 oracle, exercises active extension limiting,
compression invariance, ground-supported bending, unequal-mass fruit-pair
separation, cloth/fruit plane support, and a fruit crossing fully through a
yarn segment between frame endpoints and being returned to the correct side,
plus one yarn segment crossing another in a full substep, and requires
byte-identical replay. It also executes a three-substep rising top-seam handle
inside one command buffer, requires byte-identical trajectory replay, and
requires its final physical state to match three separately submitted
substeps exactly. This qualifies persistent state and a time-varying seam
target. The separate 480-frame pickup qualification then proves the complete
top-seam grab, spill, plural ground contact, zero final residuals, and exact
trajectory replay.
See
[docs/METAL_CLOTH.md](docs/METAL_CLOTH.md) for its exact boundary.

The remaining physical-realism limits and their required evidence are tracked
explicitly in [docs/CLOTH_REALISM_AUDIT.md](docs/CLOTH_REALISM_AUDIT.md).

An installed or relocated harness can load a specific library with
`--metallib path/to/NumiTemporalCone.metallib`.

## Numerical contract

- FP32 Metal execution with explicit finite/capacity failures.
- Deterministic fixed-capacity work queues and scan-ordered packets.
- SIMD32-native execution, with homogeneous Wave8/Wave16 cohorts when safe.
- Coupled 3x3 normal/tangent response blocks with unshifted PSD admission and
  deterministic rank-deficiency conditioning.
- Scale-safe cross-contact Cauchy-Schwarz curvature admission before
  iteration, backed by the final whole-island energy certificate.
- Per-environment failure publication; no silent contact dropping.
- Exact Euclidean projection onto isotropic, anisotropic, one-axis degenerate,
  and capped elliptic friction cones.
- Scale-homogeneous extreme-magnitude cone projection with an unchanged
  ordinary-magnitude fast path and typed rejection of unrepresentable results.
- Impulse-dimensional cone feasibility certificates matched to the absolute
  and relative convergence tolerances.
- Packed, sorted 3x3 block-CSR Delassus operators with exact dense parity.
- Stable GPU compaction of current/predicted active contacts followed by exact
  dynamic CSR construction, including a certified zero-contact island state.
- Deterministic GPU composition of `J M^-1 J^T + R` from shared-owner
  Jacobians and response columns.
- Full assembled-operator PSD admission by normalized, packed, deterministic
  semidefinite Cholesky before the streamed solver header is published.
- GPU generation of rigid 6-DOF `J` and `M^-1 J^T` directly from contact
  frames, inverse mass, and world-space inverse inertia.
- Canonical articulated kinematics and mass assembly, checked Cholesky
  factorization, and three triangular `M^-1 J^T` solves per contact without
  forming an inverse.
- Deterministic compensated articulated mass accumulation and an exact
  factor-derived `kappa_infinity` admission gate before response publication.
- A qualified seven-stage streamed inverse-ABA transaction with
  GPU-produced contact-frame right-hand sides, sparse assembly, cone solve,
  generalized-velocity publication, and a rigorous factor-free condition
  upper bound on one command buffer. ABI v3 admits this mode for fixed-root
  scalar trees with positive authored armature; the dense mode remains the
  explicit fallback for unsupported or rejected states.
- GPU-derived implicit spring-damper CFM, penetration-recovery targets, and
  thresholded restitution from versioned contact material laws.
- Canonical per-body impulse accumulation with no floating-point atomics.
- Symplectic position advance and exponential-map quaternion integration with
  whole-island rollback.
- Scale-aware KKT, cone-feasibility, and nonpositive-energy success gates.
- Runtime Delassus infinity-norm measurement and a rigorous
  `kappa_2 <= ||W||_infinity / CFM` conditioning upper gate.
- Direct particle kinetic-energy accounting against authored
  penetration-recovery work, plus measured sticking/sliding regime coverage.
- Independent post-run CPU reconstruction of bag contact geometry, free
  velocity, CSR topology, and Delassus coefficients, followed by double-
  precision KKT/cone/VI/objective evaluation and an FP64 solve from zero.
- Two-SIMD32 apply-stage reductions for node, edge, ball, energy, friction,
  escape, and warm-start evidence; no serial lane-0 scene scan.
- Independent dual-feasibility and variational-inequality complementarity
  admission, evaluated at the accepted iterate rather than inferred from an
  update norm.
- SIMD32 block-Jacobi islands with KKT-preserving scalar contact metrics.
- Delayed metric FISTA acceleration with deterministic adaptive and bounded
  restart, while only non-extrapolated iterates can satisfy the KKT gate.
- Typed nonconvergence and warm-start rollback instead of partial publication.
- PSD/SPD, nonpositive-objective, shared-rigid `1/3`, and FP64 qualification
  gates.

See [docs/MATHEMATICS.md](docs/MATHEMATICS.md) for the equations and evidence
boundary, [docs/ISLAND_SOLVER.md](docs/ISLAND_SOLVER.md) for the coupled
SIMD32 method, [docs/OPERATOR_ASSEMBLY.md](docs/OPERATOR_ASSEMBLY.md) for the
response-column producer, [docs/BRAIDED_BAG.md](docs/BRAIDED_BAG.md) for the
executable deformable containment benchmark,
[docs/CLOTH_BAG.md](docs/CLOTH_BAG.md) for the dense cloth mechanics reference,
[docs/METAL_CLOTH.md](docs/METAL_CLOTH.md) for the qualified Metal axial/grip
subset and its remaining transaction boundary, and
[docs/QUALIFICATION.md](docs/QUALIFICATION.md) for measured Apple GPU evidence.
See
[docs/RIGID_MECHANICS.md](docs/RIGID_MECHANICS.md) for the velocity-level rigid
contact path, and
[docs/ARTICULATED_MECHANICS.md](docs/ARTICULATED_MECHANICS.md) for the
factor-backed articulated path. The harnesses exercise contact-space
mathematics, rigid and articulated operator generation, velocity publication,
inverse-ABA response actions, and both complete dense and inverse-candidate
streamed solves on a real Metal device. The braided-bag harness adds a complete
interacting trajectory with GPU contact refresh and deformable point-mass
mechanics. The other harnesses remain scoped probes and do not refresh general
collision geometry or integrate articulated configuration.
