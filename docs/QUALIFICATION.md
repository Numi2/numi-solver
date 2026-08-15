# Temporal Cone qualification

## Apple M4 local-block gate

Measured on 2026-08-16 with Apple Metal toolchain `32023.883`:

```sh
./build/numi-solver-math --cases 65536 --replays 5 --iterations 16
./build/numi-solver-math --cases 65536 --replays 5 --iterations 16 --isotropic
```

The qualification gate requires:

- zero unexpected case failures;
- maximum FP32-versus-FP64 relative error at most `1e-5`;
- maximum impulse-dimensional elliptic-cone violation at most `2e-6`;
- maximum normalized fixed-point residual at most `2e-6`;
- byte-identical outputs across every replay;
- separating contact maps to zero;
- the isotropic sliding case lies on the cone;
- the authored normal cap is respected;
- either zero-friction tangent is exactly inactive while the orthogonal
  positive-friction tangent remains physical;
- a sub-epsilon nonzero inactive input is still projected to exact zero;
- the capped one-axis cone reaches its exact interval boundary;
- finite `1e20`-scale unbounded, capped, and anisotropic projections match the
  independent FP64 closest point without FP32 squared-norm overflow.

The adversarial anisotropic batch produced:

```text
cases=65536
replays=5
solver_iterations=16
failed_cases=0
max_fp64_relative_error=0.000003368
max_cone_violation=0.000000954
max_relative_fixed_point_residual=0.000000477
dimensional_violation_probe=500000.000000000
deterministic_replay=true
zero_v_axis_cone=true
zero_u_axis_cone=true
degenerate_cap=true
exact_inactive_axis=true
indefinite_local_block_rejected=true
extreme_unbounded_projection=true
extreme_capped_projection=true
extreme_anisotropic_projection=true
dimensional_cone_certificate=true
```

The dimensional certificate probe authors `lambda=(1e6,1.5e6,0)` with unit
friction. Its exact cone excess is `500000` impulse units. A radius ratio would
report only `0.5` and would be incorrectly compared with an impulse tolerance
larger than one. The measured GPU value is the exact FP32 representation of
the FP64 oracle value.

The measured isolated local-block throughput was approximately 33.5 million
problems/s for the adversarial anisotropic batch and 103.4 million problems/s
for the predominantly isotropic batch. The latter takes the closed-form fast
path; these are local mathematical blocks, not full environment steps.

## Coupled streamed SIMD32 island gate

Measured on 2026-08-16, the coupled batch mixes 1/2/4/8/16/32-contact chain,
ring, star, banded, clustered, tree, and occasional full-clique operators.
Every operator is assembled as a positive diagonal plus deterministic sparse
outer products, then independently checked by FP64 Cholesky. Coupling
strength spans `0.25` through `8.0`. The contact set mixes isotropic,
anisotropic, and one-axis degenerate cones, normal caps, separating contacts,
and nonzero warm starts.

The same FP32 operator is sent through both the fixed dense kernel and packed
3x3 block-CSR kernel. Qualification requires byte-identical impulses,
statuses, KKT residuals, objectives, iteration counts, and rollback outputs.
The independent CPU solve uses double-precision cone roots, a `1e-11 + 1e-10`
KKT gate, and up to 20,000 iterations rather than copying the GPU stop point.

Measured command:

```sh
./build/numi-solver-islands --islands 4096 --replays 5
```

Apple M4 result:

```text
islands=4096 contacts=42981 replays=5 failed_islands=0
max_fp64_impulse_error=0.000001825
max_fp64_objective_error=0.000000191
max_fp64_kkt_residual=0.000000000
max_kkt_residual=0.000001500
max_cone_violation=0.000000089
max_positive_objective=0.000000000
degenerate_cone_contacts=2184
max_degenerate_inactive_impulse=0.000000000
max_degenerate_active_impulse=0.508176446
max_iterations=105
iteration_p50=20
iteration_p95=44
iteration_p99=72
accelerated_islands=2269
max_acceleration_restarts=6
deterministic_replay=true
dense_deterministic=true
dense_stream_bitwise=true
typed_failures=true
row_bound_overflow_rejected=true
indefinite_local_block_rejected=true
cross_contact_curvature_rejected=true
positive_objective_rejected=true
deterministic_failures=true
failure_rollback=true
spd_cholesky=true
min_spd_pivot=0.894427198
shared_rigid_oracle=true
under_relaxed_path=true
average_gpu_seconds=0.005047083
islands_per_second=811557.83
contacts_per_second=8516007.58
contact_iterations_per_second=298209460.32
streamed_buffer_bytes=10126748
dense_gpu_seconds=0.011034075
dense_to_stream_speedup=2.186228012
dense_buffer_bytes=159711232
stream_to_dense_memory=0.063406611
streamed_blocks=169861
block_fill=0.182447114
max_island_blocks=1024
full_capacity_islands=17
stream_threadgroup_memory=2560
dense_threadgroup_memory=2560
stream_max_threads=1024
result=PASS
```

The streamed representation used 6.34% of the dense qualification buffers and
the isolated streamed kernel was 2.186x faster for the same byte-identical
FP32 solve. GPU time excludes CPU oracle work and buffer upload. These ratios
describe this declared topology mix; they are not universal scene claims.

The first correct one-axis implementation reused the 28-step anisotropic
bisection and measured `0.006112458` seconds on this same batch. Replacing it
with the exact two-dimensional wedge formula measured `0.004948617` seconds,
a 1.235x solver-kernel speedup with unchanged FP64, KKT, determinism, and
rollback gates. The richer batch remains within 0.4% of the prior
positive-friction-only `0.004931358`-second measurement while qualifying 2,184
lower-dimensional contacts.

Eight repeated streamed measurements after adding the curvature gate ranged
from `0.004878792` to `0.005195750` seconds, with a `0.005042454`-second
median. The prior `0.004913000`-second point remains inside that observed
run-to-run band; the added certificate retains the 2,560-byte threadgroup
footprint.

The combined metallib SHA-256 for this cross-contact-curvature milestone was
`05f15378c67c40539949c1a853b2048957157aeb6740e1b2684c3ae7f1758652`.

On the earlier 1,024-island positive-friction qualification batch, the
unaccelerated revision measured `0.004850083` seconds, 431 maximum iterations,
and p99 262. Delayed/adaptively restarted metric acceleration measured
`0.001754212` seconds, 103 maximum iterations, and p99 74: 2.76x streamed
throughput, 4.18x lower maximum iteration count, and 3.54x lower p99. The
operator, cone, tolerances, FP64 oracle, and failure gates were unchanged
within that acceleration comparison. Pipeline reflection reports 2,560 bytes
of static threadgroup memory and a 1,024-thread hardware limit; the solver
dispatch remains one SIMD32 group per island.

The typed failure batch separately verifies malformed symmetry, failed local
conditioning, bounded iteration exhaustion, invalid ABI, unsorted/duplicate
CSR, capacity overflow, an exact cone projection outside FP32 range,
finite-entry FP32 row-sum overflow, a positive-determinant local block with two
small negative eigenvalues that the CFM shift would otherwise hide, and an
impossible unit-diagonal/coupling-2 cross-contact block. A separate
three-contact equicorrelation operator has valid 2x2 principal minors but a
negative higher-order mode; its feasible zero-KKT, positive-objective point is
rejected by final energy admission. Dense and streamed outputs are poisoned
before dispatch;
deterministic failure replay, explicit zero publication for invalid state, and
projected warm-start rollback for failed final admission must still hold. The
analytic shared-rigid case verifies the coupled `(1/3, 1/3)` solution rather
than the incorrect independent-contact `(1/2, 1/2)` result.

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
max_cone_violation=0.000000060
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
non_psd_regularization_rejected=true
deterministic_failures=true
failure_rollback=true
average_gpu_seconds=0.001706492
assembly_gpu_seconds=0.000788167
assembly_fraction=0.461863757
islands_per_second=600061.53
blocks_per_second=24453093.33
assembly_blocks_per_second=52944386.60
factor_fmas_per_second=1792688124.99
contact_iterations_per_second=98769307.39
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
measurement produced 52.94 million blocks/s and the complete assembly/solve
chain produced 600,062 islands/s. One unreported warmup command precedes each
timed path. These are isolated kernel measurements, not environment-step or
energy claims.

The adversarial transaction cases independently corrupt one response column,
omit a required shared-owner block, and author a positive-determinant
regularization block with two negative eigenvalues. All are deterministically
rejected; the output stream header remains invalid and the chained solver
publishes zero impulses.

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

## Articulated response-to-generalized-velocity gate

The articulated gate imports the canonical `numisolver` articulated operator
from the same recorded source revision, then independently improves its long
mass reductions and Cholesky dot products with deterministic compensated
accumulation. For each fixed-base serial mechanism it builds world poses,
analytic point Jacobians, the generalized mass matrix, and a checked lower
Cholesky factor. The response adapter rotates each point Jacobian into a
right-handed contact frame and solves `L L^T X = J^T` for all three contact
axes without forming `M^-1`.

Five Metal encoders run on one command buffer: articulated operator, response
adapter, sparse Delassus assembly, Temporal Cone solve, and canonical
generalized-velocity publication. The CPU does not observe an intermediate.
An independent FP64 planar-mechanism model reconstructs the mass matrix,
analytic Jacobians, central finite-difference Jacobians, response columns,
Delassus blocks, contact-law target, published velocity, solve residual, and
generalized kinetic-energy budget. The configuration sweep spans wide joint
angles. The final three islands contain nonfinite operator state, an invalid
frame, and an invalid restitution law. The operator payload and complete input
velocity vector must retain their transactional values.

The dense mode of ABI v3 also reconstructs the exact FP32 infinity norm of `M` and applies one
unit solve per DoF to compute `||M^-1||_inf`. A response is not publishable
when their product exceeds `16384`, even if its backward residual is small.
The conditioning rejection gate proves this distinction with the same 32-DoF
mechanism under low authored armature.

Measured command:

```sh
./build/numi-solver-articulated --islands 1024 --replays 20
./build/numi-solver-articulated-capacity --islands 256 --replays 10
./build/numi-solver-articulated-conditioning --islands 64 --replays 10
./build/numi-solver-articulated-zero-armature --islands 8 --replays 2
```

Apple M4 result:

```text
two_link: dofs=2 islands=1024 valid=1021 contacts=2042
deterministic=yes rollback=yes failed_valid=0
mass_max_scaled_error=0.000000214
jacobian_max_scaled_error=0.000000134
finite_difference_max_abs_error=0.000001870
response_max_scaled_error=0.000000912
delassus_max_scaled_error=0.000000086
free_velocity_max_abs_error=0.000000125
publication_max_abs_error=0.000000075
gpu_response_backward_error=0.000000017
condition_infinity=26.5987854
condition_max_scaled_error=0.000000714
energy_budget_violation=0 max_iterations=47
best_gpu_chain_seconds=0.001381417
islands_per_second=741268.07 contacts_per_second=1478192.78

capacity: dofs=32 islands=256 valid=253 contacts=8096 blocks_per_island=1024
deterministic=yes rollback=yes failed_valid=0
mass_max_scaled_error=0.000002448
jacobian_max_scaled_error=0.000001215
finite_difference_max_abs_error=0.000043445
response_max_scaled_error=0.000018380
delassus_max_scaled_error=0.000000297
free_velocity_max_abs_error=0.000002601
publication_max_abs_error=0.000000283
fp64_solve_residual=0.0000000000000400
gpu_response_backward_error=0.000000000481
condition_infinity=15855.1621
condition_max_scaled_error=0.000044039
energy_budget_violation=0 max_iterations=47
best_gpu_chain_seconds=0.025609167
islands_per_second=9996.42 contacts_per_second=316136.80
operator_threadgroup_bytes=6928
factor_bytes=1048576 jacobian_response_bytes=6291456 block_bytes=9437184

conditioning_rejection: dofs=32 islands=64 rejected_valid=61
maximum_condition_infinity=182525.984 threshold=16384
deterministic=yes rollback=yes
result=PASS
```

Each timing is the best Metal command-buffer GPU timestamp across the declared
byte-identical replays. It includes all five GPU stages and excludes command
encoding, CPU oracle work, and buffer allocation. A concurrent external Metal
probe made slower samples non-isolated, so only the least-contended observed
timestamps are reported. The capacity mechanism uses explicitly authored
`0.32`-to-`0.48 kg m^2` generalized armature; this is part of its physical
operator, not hidden numerical regularization. These are mechanism/contact
workloads, not environment-step or energy-efficiency claims.
The combined metallib SHA-256 for this milestone was
`ce42170f68e599d5e12fcec4f4b235ffee34593b926e3f2d0c80c78748ff4060`.

## Streamed inverse-ABA admitted response mode

The canonical O(n) articulated inverse-mass owner is compiled with strict
FP32 arithmetic and contact-count streamed right-hand sides. The candidate
executes seven encoders on one command buffer: kinematics-only point
Jacobians, checked contact-frame preparation, inverse ABA, transactional
response finalization, sparse Delassus assembly, Temporal Cone solve, and
generalized-velocity publication. There is no CPU-visible intermediate.

For the 256-island capacity sweep, all 253 valid mechanisms and all candidate
payloads replayed byte-identically. The prepared Jacobians and streamed inverse
responses were bit-identical to the separately executed diagnostic stages, and
the response-layout transpose was bit exact. Against the independent FP64
physical model and the defining sparse/publication equations:

```text
inverse_response_max_scaled_error=0.00000104244
inverse_aba_backward_error=0.000000035886
dense_response_max_scaled_error=0.0000183799
dense_inverse_max_abs_difference=0.0000183880
candidate_delassus_max_scaled_error=0.000000315362
candidate_publication_max_abs_error=0.000000185464
candidate_energy_budget_violation=0
candidate_dense_velocity_max_abs_difference=0.0000538230
candidate_rollback=yes failed_candidate_valid=0
```

The inverse path independently certifies the fixed-root physical decomposition
`M = J_b^T I_b J_b + D_a`. Absolute world-inertia contributions plus bounded
FP32 accumulation give an upper bound for `trace(M)`, while positive authored
armature gives `lambda_min(M) >= min(D_a)`. The admitted capacity batch has
`kappa_2(M) <= 5763.177`, below the declared `16384` threshold, and the GPU
certificate never underestimates the independently reconstructed FP64
`trace(M)/min(D_a)` bound.

The low-armature adversary remains rejected by the dense gate at
`kappa_infinity=182540.469`. Inverse ABA still computes the response for
diagnosis (`0.0000120349` scaled error and `0.0000000225499` backward error),
but finalization obtains `kappa_2(M) <= 91609.625`, rejects it before assembly,
and republishes the exact input velocity. The small ABA pivot ratio (`5.43672`)
is thus correctly excluded from admission semantics.
The separate zero-armature target produces an infinite certificate and proves
the same typed rollback when no positive physical lower bound exists; its
uncertified inverse response is never assembled or published.

Apple M4 command-buffer timestamps for the 256-island capacity batch were:

```text
dense_five_stage_seconds=0.027523208
inverse_seven_stage_seconds=0.016023000
inverse_dense_speedup=1.71773128
inverse_aba_seconds=0.008794167
inverse_aba_rhs_per_second=2761830.78
factor_bytes_avoided=1048576
```

Each chain value is the least-contended Metal GPU timestamp across five
bit-identical replays and includes its complete operator-to-publication
transaction. The candidate binds only a one-float unused mass-output sentinel
to the kinematics-only ABI, so it does not allocate the 1 MiB dense factor
packet at this capacity. ABI v3 promotes the inverse mode for fixed-root scalar
trees that pass this certificate. Dense response remains the explicit path for
unsupported roots, nonpositive armature, or rejected bounds.

The combined metallib SHA-256 for this candidate milestone was
`7cd79f57810d178ebd37f99ed72f09f180de1b0329bd87cfef45937811263bf0`.

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
constant-twist free flight. The articulated gates qualify serial-chain
mass/Jacobian operators through the 32-DoF/32-contact adapter capacity,
factor-backed response columns, exact infinity-condition admission,
deterministic generalized-velocity publication, and their independent FP64,
finite-difference, residual, and energy checks. The inverse-ABA candidate
additionally qualifies the O(n) mass action, contact-frame preparation,
sparse assembly, cone solve, transactional publication, rollback, and a
state-local fixed-root condition upper bound on one command buffer. These gates
do not qualify collision
generation or refresh, articulated configuration integration, arbitrary
imported mechanisms, mechanisms above the declared adapter capacity, or a
complete interacting physical trajectory. Those remain separate layers.
