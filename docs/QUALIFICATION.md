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
max_fp64_objective_error=0.000000134
max_fp64_kkt_residual=0.000000000
max_kkt_residual=0.000001500
max_cone_violation=0.000000119
max_positive_objective=0.000000000
max_iterations=103
iteration_p50=20
iteration_p95=44
iteration_p99=74
accelerated_islands=561
max_acceleration_restarts=5
deterministic_replay=true
dense_deterministic=true
dense_stream_bitwise=true
typed_failures=true
deterministic_failures=true
failure_rollback=true
spd_cholesky=true
min_spd_pivot=0.894427198
shared_rigid_oracle=true
under_relaxed_path=true
average_gpu_seconds=0.001754212
islands_per_second=583737.72
contacts_per_second=6113854.52
contact_iterations_per_second=214952863.97
streamed_buffer_bytes=2550140
dense_gpu_seconds=0.003005421
dense_to_stream_speedup=1.713259279
dense_buffer_bytes=39927808
stream_to_dense_memory=0.063868770
streamed_blocks=42961
block_fill=0.185069033
max_island_blocks=1024
full_capacity_islands=5
stream_threadgroup_memory=2560
dense_threadgroup_memory=2560
stream_max_threads=1024
result=PASS
```

The streamed representation used 6.39% of the dense qualification buffers and
the isolated streamed kernel was 1.713x faster for the same byte-identical
FP32 solve. GPU time excludes CPU oracle work and buffer upload. These ratios
describe this declared topology mix; they are not universal scene claims.

On the same Apple M4 qualification batch, the preceding unaccelerated revision
measured `0.004850083` seconds, 431 maximum iterations, and p99 262. The
delayed/adaptively restarted metric acceleration measured `0.001754212`
seconds, 103 maximum iterations, and p99 74: 2.76x streamed throughput, 4.18x
lower maximum iteration count, and 3.54x lower p99. The operator, cone,
tolerances, FP64 oracle, and failure gates were unchanged. Pipeline reflection
reports 2,560 bytes of static threadgroup memory and a 1,024-thread hardware
limit; the solver dispatch remains one SIMD32 group per island.

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
max_iterations=27
iteration_p50=13
iteration_p95=20
iteration_p99=26
accelerated_islands=68
max_acceleration_restarts=2
deterministic_replay=true
shared_rigid_oracle=true
asymmetric_rejected=true
missing_coupling_rejected=true
deterministic_failures=true
failure_rollback=true
average_gpu_seconds=0.001729742
assembly_gpu_seconds=0.000784892
assembly_fraction=0.453762364
islands_per_second=591995.91
blocks_per_second=24124411.61
assembly_blocks_per_second=53165298.65
factor_fmas_per_second=1800168170.33
contact_iterations_per_second=97441718.03
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
measurement produced 53.17 million blocks/s and the complete assembly/solve
chain produced 591,996 islands/s. One unreported warmup command precedes each
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
linear momentum. Independent material equations reconstruct spring-damper CFM,
thresholded restitution, capped penetration recovery, and the allowable
impulse-energy budget. The final two islands contain a nonorthogonal contact
frame and invalid restitution respectively; both must fail every downstream
transaction while restoring input velocities and poses byte-for-byte.

Measured command:

```sh
./build/numi-solver-rigid --islands 1024 --replays 10
```

Apple M4 result:

```text
islands=1024 valid_bodies=2606 valid_contacts=3401 blocks=38944
operator_max_abs_error=0.000000097
free_velocity_max_abs_error=0.000000098
regularization_max_abs_error=0.000000011
publication_max_abs_error=0.000000095
pose_max_abs_error=0.000000027
quaternion_norm_max_error=0.000000054
free_flight_steps=240
free_flight_max_abs_error=0.000001955
free_flight_norm_error=0.000000051
free_flight_deterministic=yes
analytic_impulse_error=0.000000010
analytic_velocity_error=0.000000020
impact_impulse_error=0.000000048
impact_velocity_error=0.000000036
momentum_max_abs_error=0.000000036
energy_max_increase=0.000000000
energy_budget_max_violation=0.000000000
restitution_contacts=205 recovery_contacts=588
restitution_target_max=0.400000006 recovery_target_max=0.200000003
kkt_max=0.000003547
cone_max=0.000000000
iterations_max=105 p50=8 p95=53 p99=105
accelerated_islands=428 max_acceleration_restarts=3
deterministic=yes
invalid_frame_law_rollback=yes
failed_valid=0
one_command_buffer=yes
cpu_readback_between_stages=no stages=5
average_chain_seconds=0.001779808
islands_per_second=575342.85
contacts_per_second=1910879.90
result=PASS
```

The reported chain time includes all five GPU stages and excludes CPU oracle
work. The redundant 32-contact cliques remain in the timing population.
Delayed metric acceleration and deterministic adaptive restart reduce their
observed maximum from 959 iterations before this change to 105 without
loosening the KKT gate or removing the ill-conditioned cases. Complete
five-stage chain time fell from `0.013030367` to `0.001779808` seconds on the
same declared batch, a 7.32x throughput increase. The added GPU contact-law
generation is included in the latter measurement.

The combined metallib SHA-256 was
`556b4672edfb5db1cb98b4f070562880c77bc07a7abb410a97415320c7611cd6`.

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
publication, implicit spring-damper regularization, restitution/recovery
targets, their physical energy budget, one-step pose advancement, and
constant-twist free flight. It does not qualify collision generation or
refresh, articulated response, or a complete interacting physical trajectory.
Those remain separate layers.
