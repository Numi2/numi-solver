# Temporal Cone qualification

## Apple M4 local-block gate

Measured on 2026-08-15 with Apple Metal toolchain `32023.883`:

```sh
./build/numi-solver-math --cases 65536 --replays 5 --iterations 16
./build/numi-solver-math --cases 65536 --replays 5 --iterations 16 --isotropic
```

The qualification gate requires:

- zero unexpected case failures;
- maximum FP32-versus-FP64 relative error at most `1e-5`;
- maximum elliptic-cone violation at most `2e-6`;
- maximum normalized fixed-point residual at most `2e-6`;
- byte-identical outputs across every replay;
- separating contact maps to zero;
- the isotropic sliding case lies on the cone;
- the authored normal cap is respected.

The adversarial anisotropic batch produced:

```text
cases=65536
replays=5
solver_iterations=16
failed_cases=0
max_fp64_relative_error=0.000003368
max_cone_violation=0.000000119
max_relative_fixed_point_residual=0.000000477
deterministic_replay=true
```

The measured isolated local-block throughput was approximately 33.4 million
problems/s for the adversarial anisotropic batch and 103.4 million problems/s
for the predominantly isotropic batch. The latter takes the closed-form fast
path; these are local mathematical blocks, not full environment steps.

## Coupled SIMD32 island gate

The dense coupled-island qualification uses symmetric positive-definite
operators with rank-one shared modes spanning coupling strengths from `0.08`
through `4.0`. It mixes 1/2/4/8/16/32-contact islands, isotropic and
anisotropic cones, normal caps, separating contacts, and nonzero warm starts.

Measured command:

```sh
./build/numi-solver-islands --islands 1024 --replays 5
```

Apple M4 result:

```text
islands=1024
contacts=10725
replays=5
failed_islands=0
max_fp64_impulse_error=0.000001523
max_fp64_objective_error=0.000000154
max_natural_residual=0.000001498
max_cone_violation=0.000001073
max_iterations=251
deterministic_replay=true
typed_failures=true
deterministic_failures=true
failure_rollback=true
average_gpu_seconds=0.013931233
islands_per_second=73503.90
contacts_per_second=769852.87
contact_iterations_per_second=98542746.81
buffer_bytes=39927808
result=PASS
```

The typed failure batch separately verifies asymmetric-operator rejection,
local factorization failure, bounded iteration exhaustion, deterministic
failure replay, and warm-start rollback. The measured GPU interval covers the
island kernel, not CPU oracle work or buffer upload. The 39,927,808-byte shared
allocation is the deliberately dense qualification representation, not a
sparse-production memory claim.

The combined metallib SHA-256 was
`c1807eaa8c1e0915e2e3b946a5dda11a293464ea472e876c45475e229fb38edf`.

## Evidence boundary

These measurements execute the real conditioned inverse and cone projection
helpers from `MetalWorldContact.metal` on Apple GPU. The FP64 implementation is
independent and solves anisotropic/capped scalar roots to double precision.

This is strong evidence for local and dense coupled contact-space mathematics,
FP32 stability, deterministic SIMD32 convergence, rollback, and isolated
kernel cost. It is not evidence for collision generation, sparse/streamed
operator equivalence, rigid/articulated velocity publication, integration, or
a complete physical trajectory. Those remain separate qualification layers.
