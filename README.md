# Numi Solver

Numi Solver is the independent Apple Metal home of the Temporal Cone contact
solver extracted from Numi Lab's `numisolver` branch.

This repository intentionally contains only the owning Metal solver source and
its pointer-free GPU ABI headers. It does not contain Numi Lab's robot models,
training runtime, tasks, applications, assets, research artifacts, rendering,
or website.

## Solver boundary

`src/metal/MetalWorldContact.metal` is preserved byte-for-byte from the source
revision recorded in [PROVENANCE.md](PROVENANCE.md). The file owns the complete
contact pipeline required by Temporal Cone, including deterministic island and
tile construction, Wave8/16/32 cohort selection, coupled normal/tangent cone
updates, distributed-island reduction, stiff-island ordered replay, warm-start
publication, and transactional status reduction.

The extracted kernel still speaks the original versioned MetalWorld ABI. This
keeps numerical behavior and integration contracts intact while the solver is
developed independently. A small standalone host API is a future boundary; it
has not been invented during extraction.

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
`-fno-fast-math`.

## Numerical contract

- FP32 Metal execution with explicit finite/capacity failures.
- Deterministic fixed-capacity work queues and scan-ordered packets.
- SIMD32-native execution, with homogeneous Wave8/Wave16 cohorts when safe.
- Coupled 3x3 normal/tangent response blocks with deterministic conditioning.
- Per-environment failure publication; no silent contact dropping.
- Exact elliptic friction-cone projection and residual reporting.

Building the metallib proves source/toolchain compatibility. It does not by
itself prove a complete physical step, deterministic replay, or a hardware
outcome; those require a host integration and executable physics probe.

