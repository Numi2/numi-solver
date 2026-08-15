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

## Coupled streamed SIMD32 island gate

Measured on 2026-08-15, the coupled batch mixes 1/2/4/8/16/32-contact chain,
ring, star, banded, clustered, tree, and occasional full-clique operators.
Every operator is assembled as a positive diagonal plus deterministic sparse
outer products, then independently checked by FP64 Cholesky. Coupling
strength spans `0.25` through `8.0`. The contact set mixes isotropic and
anisotropic cones, normal caps, separating contacts, and nonzero warm starts.

The same FP32 operator is sent through both the fixed dense kernel and packed
3x3 block-CSR kernel. Qualification requires byte-identical impulses,
statuses, KKT residuals, objectives, iteration counts, and rollback outputs.
The independent CPU solve uses double-precision cone roots, a `1e-11 + 1e-10`
KKT gate, and up to 20,000 iterations rather than copying the GPU stop point.

Measured command:

```sh
./build/numi-solver-islands --islands 1024 --replays 10
```

Apple M4 result:

```text
islands=1024 contacts=10725 replays=10 failed_islands=0
max_fp64_impulse_error=0.000001834
max_fp64_objective_error=0.000000150
max_fp64_kkt_residual=0.000000000
max_kkt_residual=0.000001500
max_cone_violation=0.000000119
max_positive_objective=0.000000000
max_iterations=431
iteration_p50=20
iteration_p95=94
iteration_p99=262
deterministic_replay=true
dense_deterministic=true
dense_stream_bitwise=true
typed_failures=true
deterministic_failures=true
failure_rollback=true
spd_cholesky=true
min_spd_pivot=0.894427198
shared_rigid_oracle=true
average_gpu_seconds=0.004718708
islands_per_second=217008.54
contacts_per_second=2272867.75
contact_iterations_per_second=163914983.56
streamed_buffer_bytes=2550140
dense_gpu_seconds=0.008276950
dense_to_stream_speedup=1.754071110
dense_buffer_bytes=39927808
stream_to_dense_memory=0.063868770
streamed_blocks=42961
block_fill=0.185069033
max_island_blocks=1024
full_capacity_islands=5
result=PASS
```

The streamed representation used 6.39% of the dense qualification buffers and
the isolated streamed kernel was 1.754x faster for the same byte-identical
FP32 solve. GPU time excludes CPU oracle work and buffer upload. These ratios
describe this declared topology mix; they are not universal scene claims.

The typed failure batch separately verifies malformed symmetry, failed local
conditioning, bounded iteration exhaustion, invalid ABI, unsorted/duplicate
CSR, capacity overflow, nonfinite FP32 warm-start arithmetic, deterministic
failure replay, zero publication for invalid state, and projected warm-start
rollback for nonconvergence. The analytic shared-rigid case verifies the
coupled `(1/3, 1/3)` solution rather than the incorrect independent-contact
`(1/2, 1/2)` result.

The combined metallib SHA-256 was
`f218910754929afb6cc1e10cea7574db0777f7c81f837c790de5015d1ada27e7`.

## Evidence boundary

These measurements execute the real conditioned inverse and cone projection
helpers from `MetalWorldContact.metal` on Apple GPU. The FP64 implementation is
independent and solves anisotropic/capped scalar roots to double precision.

This is strong evidence for local cone mathematics, coupled contact-space KKT,
FP32 stability, deterministic SIMD32 convergence, dense-versus-streamed
operator equivalence, transaction rollback, SPD/objective checks, and isolated
kernel cost. It qualifies the streamed operator consumer, not construction of
`J M^-1 J^T` from collision Jacobians or articulated response columns. Contact
generation, operator production, rigid/articulated velocity publication,
integration, and a complete physical trajectory remain separate layers.
