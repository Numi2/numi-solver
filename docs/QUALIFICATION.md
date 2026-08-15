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
average_gpu_seconds=0.004850083
islands_per_second=211130.39
contacts_per_second=2211302.20
contact_iterations_per_second=159474991.07
streamed_buffer_bytes=2550140
dense_gpu_seconds=0.008173462
dense_to_stream_speedup=1.685221035
dense_buffer_bytes=39927808
stream_to_dense_memory=0.063868770
streamed_blocks=42961
block_fill=0.185069033
max_island_blocks=1024
full_capacity_islands=5
result=PASS
```

The streamed representation used 6.39% of the dense qualification buffers and
the isolated streamed kernel was 1.685x faster for the same byte-identical
FP32 solve. GPU time excludes CPU oracle work and buffer upload. These ratios
describe this declared topology mix; they are not universal scene claims.

The typed failure batch separately verifies malformed symmetry, failed local
conditioning, bounded iteration exhaustion, invalid ABI, unsorted/duplicate
CSR, capacity overflow, nonfinite FP32 warm-start arithmetic, deterministic
failure replay, zero publication for invalid state, and projected warm-start
rollback for nonconvergence. The analytic shared-rigid case verifies the
coupled `(1/3, 1/3)` solution rather than the incorrect independent-contact
`(1/2, 1/2)` result.

## Response-column assembly and chained solve gate

The assembly gate constructs sparse contact operators from packed
shared-owner Jacobians and response columns. It includes one-to-32-contact
islands, one-to-32 owner terms per contact, one-to-32 DOFs per generated
term, sparse graphs, and five full 1,024-block cliques. An independent CPU
path reconstructs the dense operator and applies FP64 Cholesky.

The assembly encoder commits a streamed header only after topology,
finiteness, and symmetry validation. The solver encoder consumes that header
on the same command buffer, with no CPU readback between stages.

Measured command:

```sh
./build/numi-solver-assembly --islands 1024 --replays 10
```

Apple M4 result:

```text
islands=1024 contacts=10725 blocks=41729 terms=41727 replays=10
failed_islands=0
max_assembly_error=0.000000000
max_symmetry_error=0.000000000
min_spd_pivot=0.905538510
max_kkt_residual=0.000001491
max_cone_violation=0.000000119
max_positive_objective=0.000000000
max_iterations=35
iteration_p50=13
iteration_p95=20
iteration_p99=30
deterministic_replay=true
shared_rigid_oracle=true
asymmetric_rejected=true
missing_coupling_rejected=true
deterministic_failures=true
failure_rollback=true
average_gpu_seconds=0.001726683
assembly_gpu_seconds=0.000767038
assembly_fraction=0.444225927
islands_per_second=593044.47
blocks_per_second=24167141.23
assembly_blocks_per_second=54402815.65
factor_fmas_per_second=1842070290.26
contact_iterations_per_second=99759461.74
factor_bytes=3505020
streamed_operator_bytes=1716156
dense_operator_bytes=37748736
factor_stream_to_dense=0.138313929
max_terms_per_contact=32
max_dofs_per_term=32
max_island_blocks=1024
full_capacity_islands=5
same_command_buffer=true
cpu_readback_between_stages=false
result=PASS
```

Factor inputs plus the assembled sparse operator used 13.83% of the fixed
dense operator storage for this declared topology mix. The assembly-only
measurement produced 54.40 million blocks/s and the complete assembly/solve
chain produced 593,044 islands/s. One unreported warmup command precedes each
timed path. These are isolated kernel measurements, not environment-step or
energy claims.

The adversarial transaction cases independently corrupt one response column
and omit a required shared-owner block. Both are deterministically rejected;
the output stream header remains invalid and the chained solver publishes zero
impulses.

## Rigid response-to-velocity gate

The rigid gate supplies body linear/angular velocities, positive inverse
masses, world-space SPD inverse inertias, contact-point offsets, and
right-handed contact frames. It chains rigid `J`/`M^-1 J^T` generation, sparse
assembly, cone solve, deterministic velocity publication, and SE(3) pose
integration in five encoders on one command buffer.

The batch mixes dynamic-static offset contacts, frictional contacts,
dynamic-dynamic pairs, eight-body chains, redundant shared-body contacts, and
32-contact full-capacity shared-body cliques. Independent CPU equations
reconstruct free contact velocity, every present Delassus coefficient, and
every published linear/angular velocity. Dynamic-only islands check total
linear momentum, and all zero-bias inelastic cases reject kinetic-energy
increase. The final island has a deliberately nonorthogonal contact frame and
must fail every downstream transaction while restoring input velocities
byte-for-byte.

Measured command:

```sh
./build/numi-solver-rigid --islands 1024 --replays 10
```

Apple M4 result:

```text
islands=1024 valid_bodies=2607 valid_contacts=3402 blocks=38944
operator_max_abs_error=0.000000098
free_velocity_max_abs_error=0.000000053
publication_max_abs_error=0.000000092
pose_max_abs_error=0.000000030
quaternion_norm_max_error=0.000000037
free_flight_steps=240
free_flight_max_abs_error=0.000001955
free_flight_norm_error=0.000000051
free_flight_deterministic=yes
analytic_impulse_error=0.000000010
analytic_velocity_error=0.000000020
momentum_max_abs_error=0.000000037
energy_max_increase=0.000000000
kkt_max=0.000003955
cone_max=0.000000000
iterations_max=959 p50=8 p95=246 p99=959
deterministic=yes
invalid_frame_rollback=yes
failed_valid=0
one_command_buffer=yes
cpu_readback_between_stages=no stages=5
average_chain_seconds=0.013030367
islands_per_second=78585.66
contacts_per_second=261082.45
result=PASS
```

The reported chain time includes all five GPU stages and excludes CPU oracle
work. The high iteration tail is retained: the redundant 32-contact cliques
exercise a deliberately less-conditioned shared response rather than being
removed from the timing population.

The combined metallib SHA-256 was
`3859c1a704419becc7f0df7a4decc44d9e725badcda08f59693161680b961192`.

## Evidence boundary

These measurements execute the real conditioned inverse and cone projection
helpers from `MetalWorldContact.metal` on Apple GPU. The FP64 implementation is
independent and solves anisotropic/capped scalar roots to double precision.

This is strong evidence for local cone mathematics, coupled contact-space KKT,
FP32 stability, deterministic SIMD32 convergence, dense-versus-streamed
operator equivalence, transaction rollback, SPD/objective checks, and isolated
kernel cost. It also qualifies numerical construction of `J M^-1 J^T + R`
from supplied Jacobians and response columns on the solver command-buffer
timeline. The rigid gate additionally qualifies contact-frame Jacobian
construction, rigid `M^-1 J^T`, deterministic linear/angular velocity
publication, one-step pose advancement, and constant-twist free flight. It
does not qualify collision generation or refresh, articulated response, or a
complete interacting physical trajectory. Those remain separate layers.
