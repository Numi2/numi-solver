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

The solver still speaks the original versioned MetalWorld ABI. A narrow probe
ABI exposes only its local 3x3 response conditioning and exact elliptic-cone
projection for direct mathematical work. The probe calls the production Metal
helpers; it does not carry a second shader implementation.

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
`-fno-fast-math`, plus three native harnesses:

- `build/numi-solver-math` for isolated local cone blocks;
- `build/numi-solver-islands` for dense-versus-streamed coupled
  1/2/4/8/16/32-contact islands;
- `build/numi-solver-assembly` for GPU response-column assembly followed by
  a streamed solve on one command buffer.

Run the default FP64 comparisons and deterministic replays:

```sh
./build/numi-solver-math
./build/numi-solver-islands
./build/numi-solver-assembly
ctest --test-dir build --output-on-failure
```

The local harness accepts `--cases N --replays N --iterations N` and
`--isotropic`. The island and assembly harnesses accept `--islands N` and
`--replays N`. Together they check separating, impact, sticking, sliding,
anisotropic friction, near-rank-deficient response, capped impulse, polar
boundary, zero axis, extreme scale, sparse topology, full block capacity,
shared response, missing coupling, and response asymmetry.

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
- Scale-aware KKT gradient-mapping convergence residuals.
- SIMD32 block-Jacobi islands with KKT-preserving scalar contact metrics.
- Typed nonconvergence and warm-start rollback instead of partial publication.
- SPD, nonpositive-objective, shared-rigid `1/3`, and FP64 qualification gates.

See [docs/MATHEMATICS.md](docs/MATHEMATICS.md) for the equations and evidence
boundary, [docs/ISLAND_SOLVER.md](docs/ISLAND_SOLVER.md) for the coupled
SIMD32 method, [docs/OPERATOR_ASSEMBLY.md](docs/OPERATOR_ASSEMBLY.md) for the
response-column producer, and [docs/QUALIFICATION.md](docs/QUALIFICATION.md)
for measured Apple GPU evidence. The harnesses exercise contact-space
mathematics, operator assembly, and the streamed solver on a real Metal
device. They do not yet generate collision Jacobians, compute upstream
rigid/articulated response columns, publish velocity updates, or integrate a
complete physical trajectory.
