# Numi Solver

Numi Solver is the independent Apple Metal home of the Temporal Cone contact
solver extracted from Numi Lab's `numisolver` branch.

This repository intentionally contains only the owning Metal solver source and
its pointer-free GPU ABI headers. It does not contain Numi Lab's robot models,
training runtime, tasks, applications, assets, research artifacts, rendering,
or website.

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
cmake -S . -B build -G Ninja
cmake --build build
```

The build produces `build/shaders/NumiTemporalCone.metallib` using `-O3` and
`-fno-fast-math`, plus five native harnesses:

- `build/numi-solver-math` for isolated local cone blocks;
- `build/numi-solver-islands` for dense-versus-streamed coupled
  1/2/4/8/16/32-contact islands;
- `build/numi-solver-assembly` for GPU response-column assembly followed by
  a streamed solve on one command buffer.
- `build/numi-solver-rigid` for rigid contact frames and mass/inertia through
  deterministic velocity publication on one command buffer.
- `build/numi-solver-articulated` for canonical articulated mass/Jacobian
  factorization, response-column solves, cone contact, and deterministic
  generalized-velocity publication on one command buffer.

Run the default FP64 comparisons and deterministic replays:

```sh
./build/numi-solver-math
./build/numi-solver-islands
./build/numi-solver-assembly
./build/numi-solver-rigid
./build/numi-solver-articulated
ctest --test-dir build --output-on-failure
```

The local harness accepts `--cases N --replays N --iterations N` and
`--isotropic`. The island, assembly, rigid, and articulated harnesses accept
`--islands N` and `--replays N`. Together they check separating, impact, sticking, sliding,
anisotropic friction, near-rank-deficient response, capped impulse, polar
boundary, zero axis, extreme scale, sparse topology, full block capacity,
shared response, missing coupling, response asymmetry, rigid momentum and
kinetic-energy budgets, implicit contact-law regularization, thresholded
restitution, penetration recovery, contact-frame validity, and transactional
velocity rollback. The articulated gate additionally checks analytic and
finite-difference two-link Jacobians, an independent FP64 mass matrix,
factor-solved response columns, conditioning diagnostics, generalized kinetic
energy, and invalid frame/material rollback.

An installed or relocated harness can load a specific library with
`--metallib path/to/NumiTemporalCone.metallib`.

## Numerical contract

- FP32 Metal execution with explicit finite/capacity failures.
- Deterministic fixed-capacity work queues and scan-ordered packets.
- SIMD32-native execution, with homogeneous Wave8/Wave16 cohorts when safe.
- Coupled 3x3 normal/tangent response blocks with deterministic conditioning.
- Per-environment failure publication; no silent contact dropping.
- Exact Euclidean projection onto isotropic, anisotropic and capped elliptic
  friction cones.
- Packed, sorted 3x3 block-CSR Delassus operators with exact dense parity.
- Deterministic GPU composition of `J M^-1 J^T + R` from shared-owner
  Jacobians and response columns.
- GPU generation of rigid 6-DOF `J` and `M^-1 J^T` directly from contact
  frames, inverse mass, and world-space inverse inertia.
- Canonical articulated kinematics and mass assembly, checked Cholesky
  factorization, and three triangular `M^-1 J^T` solves per contact without
  forming an inverse.
- GPU-derived implicit spring-damper CFM, penetration-recovery targets, and
  thresholded restitution from versioned contact material laws.
- Canonical per-body impulse accumulation with no floating-point atomics.
- Symplectic position advance and exponential-map quaternion integration with
  whole-island rollback.
- Scale-aware KKT gradient-mapping convergence residuals.
- SIMD32 block-Jacobi islands with KKT-preserving scalar contact metrics.
- Delayed metric FISTA acceleration with deterministic adaptive and bounded
  restart, while only non-extrapolated iterates can satisfy the KKT gate.
- Typed nonconvergence and warm-start rollback instead of partial publication.
- SPD, nonpositive-objective, shared-rigid `1/3`, and FP64 qualification gates.

See [docs/MATHEMATICS.md](docs/MATHEMATICS.md) for the equations and evidence
boundary, [docs/ISLAND_SOLVER.md](docs/ISLAND_SOLVER.md) for the coupled
SIMD32 method, [docs/OPERATOR_ASSEMBLY.md](docs/OPERATOR_ASSEMBLY.md) for the
response-column producer, and [docs/QUALIFICATION.md](docs/QUALIFICATION.md)
for measured Apple GPU evidence. See
[docs/RIGID_MECHANICS.md](docs/RIGID_MECHANICS.md) for the velocity-level rigid
contact path, and
[docs/ARTICULATED_MECHANICS.md](docs/ARTICULATED_MECHANICS.md) for the
factor-backed articulated path. The harnesses exercise contact-space
mathematics, rigid and articulated operator generation, velocity publication,
and the streamed solver on a real Metal device. They do not yet perform
collision detection, refresh contact geometry, integrate articulated
configuration, or execute a complete interacting physical trajectory.
