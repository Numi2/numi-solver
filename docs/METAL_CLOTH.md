# Metal cloth, seam grip, contact, and friction

## Qualified scope

`numi-solver-cloth-metal` is the first device-owned transaction for the
explicit-yarn produce-bag topology. It is not the complete bag trajectory. The
versioned ABI in `include/numi/cloth_bag_gpu.h` owns:

- 1,465 massive cloth particles;
- 2,904 axial warp, weft, and bottom constraints;
- 1,369 compliant crossing-angle knot constraints;
- 2,834 three-knot yarn-bend constraints;
- twelve fruits with translational, angular, orientation, and contact state;
- all 66 graph-colored fruit-pair candidates;
- all 34,848 fruit/yarn candidates with present-time closest geometry and
  swept conservative-advancement CCD;
- swept and present-time sphere/yarn normal position response with accumulated
  normal impulse publication;
- all 4,149,792 nonlocal yarn/yarn capsule candidates after graph-derived
  two-ring exclusion, with swept CCD and present-time normal response;
- unilateral cloth/ground and fruit/ground projection;
- the ten finite-compliance top-seam grips;
- gravity prediction and symplectic position advance;
- XPBD axial-yarn projection;
- the unilateral 28.5% extension ceiling;
- deterministic contact/strain reconciliation after that ceiling; and
- velocity publication from accepted positions;
- maximum-dissipation Coulomb friction for cloth/ground, yarn/yarn,
  fruit/yarn, fruit/fruit, and fruit/ground contact; and
- normal-load-capped fruit rolling resistance; and
- normalized fruit-orientation integration after contact impulses;
- cylinder crossflow and axial skin-friction air loads on every yarn segment;
  and
- exact quadratic translational and rotational air decay for every fruit; and
- topology-aware release classification through the ordered 48-knot mouth.

The distance, knot, bend, and fruit-pair graphs are greedily colored once into
five, six, five, and fifteen deterministic batches. No two constraints in a
batch write the same state. Each batch therefore runs in parallel without
float atomics or a lane-zero serial loop, while ordered dispatches preserve a
deterministic Gauss-Seidel schedule across colors. The ten grips touch ten
distinct seam particles and also execute in parallel. Bend projection includes
the CPU reference's active-set response for a yarn endpoint supported by the
ground.

Fruit/yarn response reuses the five conflict-free edge batches. For a batch of
`M` disjoint yarn segments, twelve threads own the twelve fruits. Phase `p`
assigns fruit `f` segment `(p+f) mod M`; a device barrier separates phases.
Every phase therefore has unique fruit and yarn-knot writers, so the complete
34,848-candidate sweep needs five ordered dispatches without float atomics or
a lane-zero loop. The FP64 oracle executes this same declared schedule.

Yarn self-contact is compiled once into 29,263 deterministic ownership-safe
batches, each containing at most 256 four-knot segment pairs. The current-state
broadphase assigns all 2,904 segment midpoints to guarded `0.1 m` cells, sorts
4,096 padded `(cell,segment)` entries with one deterministic Metal bitonic
transaction, searches the 27 neighboring cells, and maps surviving pairs into
the static schedule through a packed upper-triangular lookup. Each static pair
stores two 32-bit segment indices; the four particle endpoints are resolved
from the owning distance table instead of being duplicated. A device prefix
transaction preserves batch order while compacting only active batches for
response. The swept pass retains exhaustive AABB admission before conservative
advancement so fast relative motion cannot skip cells. No position update uses
a float atomic.

Every accepted yarn/yarn normal response appends one compact 48-byte impulse
record. A deterministic 4,096-key bitonic transaction sorts those records by
the static pair index, then one owner aggregates normal impulse and
impulse-weighted endpoint coordinates for each contacted pair. The existing
packed pair map and ownership-safe static batches are reused for the
maximum-dissipation tangential solve. This gives all four yarn endpoints one
conflict-free writer per batch without allocating a dense response record for
all 4,149,792 candidates.

For distance constraint `ij`, the device evaluates the same XPBD equation as
the FP64 cloth reference:

```math
C=\|x_j-x_i\|-\ell_0,\qquad
\alpha=\frac{c}{\Delta t^2},
```

```math
\Delta\lambda=\frac{-C-\alpha\lambda}
{w_i+w_j+\alpha}.
```

For one seam particle and virtual-handle target `g+o`, the vector attachment
is

```math
\Delta\boldsymbol\lambda=
\frac{-(x-g-o)-\alpha\boldsymbol\lambda}{w+\alpha}.
```

The post-solve extension limiter is unilateral: it corrects only
`length > 1.285 restLength`. Compression passes unchanged.

Because a strain projection can reintroduce contact overlap, the transaction
does not publish immediately after the limiter. It executes eight fixed
fruit/yarn/strain/ground reconciliation passes, one endpoint yarn-self/strain
pass, two final fruit/yarn/ground passes, then eight fixed certificate passes
that repeat endpoint self/strain followed by the two final contact passes. The
FP64 oracle uses the identical declared order. This schedule has no host
readback or data-dependent dispatch count.

For overlapping fruits `i,j`, the positional normal solve is

```math
\lambda=\frac{r_i+r_j-\|x_j-x_i\|}{w_i+w_j},\qquad
x_i\mathrel{-}=w_i\lambda n,\quad
x_j\mathrel{+}=w_j\lambda n.
```

The accumulated normal impulse is `lambda / dt`. Ground projection clamps
cloth to the yarn radius and fruit centers to their sphere radius, recording
the removed fruit advance as normal impulse.

For every fruit center `b` and moving yarn segment `[a,c]`, one Metal thread
writes one deterministic candidate record. Present-time geometry uses

```math
s=\operatorname{clamp}\left(\frac{(b-a)\cdot(c-a)}{\|c-a\|^2},0,1\right),
\qquad q=a+s(c-a),
```

with signed surface separation `||q-b||-(r_f+r_y)`. Swept geometry linearly
interpolates the fruit and both yarn endpoints and applies the CPU reference's
80-step conservative advancement. A crossing is published only when the
remaining relative normal advance after impact is positive. This stage builds
the complete deterministic contact set. The accepted present or swept normal
correction then applies the same inverse-mass and ground-aware active-set
response as the CPU reference. Normal impulse is `lambda/dt`.

After accepted positions publish velocity, each qualified contact applies the
CPU reference's maximum-dissipation tangential impulse

```math
j_t=\min\left(\frac{\|v_t\|}{d_t},\mu j_n\right),
```

where `d_t` includes translational, solid-sphere rotational, interpolated yarn,
and ground active-set response as applicable. Fruit rolling resistance applies

```math
j_r=\min\left(\frac{\|\omega_{xy}\|}{I^{-1}},
\mu_r r j_n\right).
```

The same graph-colored pair and round-robin fruit/yarn schedules prevent shared
velocity writers without float atomics. Cloth and fruit ground response are
one-thread-per-body transactions. Positive-float atomic maxima publish only
cone certificates and never own simulated state.

Air loads run between gravity and position advance, matching the FP64
transaction order. Conflict-free distance colors accumulate each segment's
force into its two endpoints. One fixed 256-lane tree reduction computes the
global dissipative attenuation from drag power and inverse-mass-weighted force
norm; a parallel particle pass then applies the force. Fruit translation and
rotation use the reference's exact one-step quadratic attenuation. No air-load
transaction can reverse relative velocity or add relative kinetic energy.

## Independent qualification

The harness reconstructs the reference bag's 48 by 28 wall and 13 by 13
closed bottom, verifies the exact particle, distance, knot, bend, grip, fruit,
pair, and 34,848 fruit/yarn candidate counts, and rejects any graph color that
shares written state. It also checks all 4,149,792 nonlocal self-pairs and
every self-contact batch for unique knot writers. It executes 32
internal-constraint/grip/contact
iterations, three extension-limit sweeps, and the fixed reconciliation and
certificate schedule on Metal, then compares every particle, fruit, contact,
and multiplier with an independent FP64 implementation of the same colored
schedule. Two GPU runs must be byte-identical. Published yarn/fruit, yarn/yarn,
ground, and strain residuals are independently measured after the last pass.

A separate transaction check authors three rising top-seam handle targets and
encodes all three full physics substeps into one command buffer. Cloth, fruit,
constraint, contact, release, and friction state stays device-resident between
substeps. Two persistent executions must be byte-identical, and the final
physical buffers must exactly match the same three substeps submitted as three
command buffers. The check also gates positive seam lift and finite attachment
lag. This is a persistence and moving-handle certificate, not a complete Metal
pickup or spill trajectory.

The focused strain case starts one segment with `0.215 m` excess extension and
another in compression. One Metal sweep must remove the extension violation,
leave the compressed length unchanged, preserve the stretched pair's center
of mass, and match the FP64 oracle. A separate ground-bend case requires a
supported endpoint to remain at the yarn radius while the free endpoint moves,
again matching an independent FP64 active-set solve. A deliberately
overlapping unequal-mass fruit pair must separate to the exact combined radius
while preserving center of mass and normal impulse. A separate ground case
must lift both a yarn particle and fruit to their physical radii and match the
FP64 normal impulse. A focused tunneling case moves a fruit completely through
a fixed segment within one 0.01-second substep; Metal must publish the same
impact time, normal, segment weight, removable advance, final separation, and
normal impulse as FP64 despite no endpoint-time crossing test being sufficient.
A second focused CCD case moves one yarn segment completely through another in
one substep while sliding tangentially. Metal must stop it at the `8 mm`
combined capsule diameter, reduce tangential slip and kinetic energy, conserve
linear momentum, and match FP64 positions, velocities, accepted swept-contact
count, friction count, and cone ratio.
The fruit-pair probe also drives tangential slip and requires slip and kinetic
energy to fall while linear momentum is conserved. The ground probe requires
cloth slip, fruit slip, and rolling speed all to fall under their independent
Coulomb and rolling caps. The sphere/yarn tunneling probe adds tangential motion
and must reduce contact slip after the CCD response. All friction velocities,
angular velocities, contact counts, and maximum cone ratios are compared with
the independent FP64 schedule. Fruit orientations are compared componentwise
with FP64 after friction, must remain unit length, and a focused tangential
fruit collision must produce a nonzero orientation change.
Separate air probes require yarn and fruit energy to fall, compare forces,
torque, linear velocity, and angular velocity with FP64, verify subdivision
invariance for a uniformly moving yarn, and require a yarn moving with the air
to receive no physical drag.
The mouth probe independently checks contained, complete cap crossing,
far-outside, grazing, edge-clear, prior-candidate edge exit, and rigidly
rotated openings. Candidate and released masks must match FP64 exactly; full
clearance must match within `2 um`.

Run it with:

```sh
./build/numi-solver-cloth-metal \
  --replays 2 --iterations 32 --strain-sweeps 3
```

The complete pickup is an explicit long-running qualification. It advances
48 device-resident substeps per 1/120-second frame, follows the 240-frame
pickup motion, then holds the final handle for a 60-frame settling tail. It
publishes only at frame boundaries, exports replay-one states every ten frames,
and compares all 300 frame hashes plus the final physical buffers with replay
two. At least two released fruit must physically rest on the plane at the end:

```sh
./build/numi-solver-cloth-metal \
  --replays 2 --iterations 32 --strain-sweeps 3 \
  --pickup-prefix build/metal-pickup --pickup-steps 300 \
  --pickup-dump-every 10
```

The Apple M4 result measured on 2026-08-21 was:

```text
abi=10 particles=1465 distances=2904 grips=10 knots=1369 bends=2834
fruits=12 fruit_pairs=66 fruit_yarn_candidates=34848
distance_colors=5 knot_colors=6 bend_colors=5 fruit_pair_colors=15
self_pairs=4149792 self_batches=29263 max_self_batch=256
failure_flags=0 deterministic=true
max_position_error=0.000000555473
max_velocity_error=0.003199523975
max_distance_lambda_error=0.000000000022
max_grip_lambda_error=0.000000000009
max_knot_lambda_error=0.000000000001
max_bend_lambda_error=0.000000000008
max_fruit_position_error=0.000000076410
max_fruit_velocity_error=0.000439966229
max_fruit_angular_velocity_error=0.000007092363
max_fruit_orientation_error=0.000000001746
max_fruit_orientation_norm_error=0.000000001746
max_yarn_aerodynamic_force=0.000000001152 expected=0.000000001152
max_fruit_aerodynamic_force=0.000000014756 expected=0.000000014756
mouth_candidate_mask=3840 expected_candidate_mask=3840
released_mask=0 expected_released_mask=0
seam_trajectory_substeps=3 replay_exact=true split_exact=true
average_lift=0.008440265059 maximum_handle_lag=0.001929601793
failure_flags=0
max_fruit_pair_penetration=0.000000000000
yarn_identity_exact=true yarn_control_qualified=true
current_yarn_overlaps=0 swept_yarn_impacts=0
max_yarn_separation_error=0.000000563364
max_yarn_current_normal_error=0.000056231269
max_active_yarn_weight_error=0.000004242114
yarn_response_count_exact=false
accepted_yarn_responses=113 expected_yarn_responses=109
response_count_mismatches=8
max_yarn_normal_impulse=0.003789614420
max_yarn_response_error=0.000000053625
max_yarn_penetration=0.000000000000
present_self_contacts=124 expected_present_self_contacts=127
swept_self_contacts=0 expected_swept_self_contacts=0
max_self_correction=0.003799978178
max_self_correction_error=0.000000002372
final_self_penetration=0.000000009758
final_ground_penetration=0.000000000000
max_strain_violation=0.000000000000
max_displacement=0.006799975103
grip_force=153.876787384189
reconciliation_passes=8 final_contact_passes=2 certificate_passes=8
friction_contacts_pair=0 expected_pair=0
friction_contacts_yarn=12 expected_yarn=10 max_count_difference=2
friction_contacts_cloth_ground=0 friction_contacts_fruit_ground=0
friction_contacts_self=7 expected_self=6 self_count_difference=1
max_friction_cone_ratio=1.000000000000 expected=1.000000000000
max_rolling_ratio=0.000000000000 expected_rolling=0.000000000000
strain_probe_initial_violation=0.215000003576
strain_probe_final_violation=0.000000007451
compression_length_change=0.000000000000
strain_probe_com_error=0.000000014901
ground_bend_position_error=0.000000056670
supported_height=0.004000000190
free_endpoint_rise=0.009750500321
fruit_pair_probe_position_error=0.000000044763
separation=1.999999962672 center_error=0.000000019868
normal_impulse=16.656669616699 friction_contacts=1
orientation_error=0.000000011760 orientation_change=0.007141246926
cone_ratio=0.057165719569 slip_before=2.999400344361
slip_after=0.000000173872 momentum_error=0.000000834465
ground_contact_position_error=0.000000000000
ground_contact_impulse_error=0.000000558794
cloth_height=0.004000000190 fruit_height=1.000000000000
fruit_normal_impulse=25.000000000000
cloth_friction_contacts=1 fruit_friction_contacts=1 rolling_contacts=1
cone_ratio=1.000000000000 rolling_ratio=1.000000000000
cloth_slip_before=1.000000000000 cloth_slip_after=0.819999992847
fruit_slip_before=3.605551275464 fruit_slip_after=1.875000083742
rolling_speed_before=2.000000000000 rolling_speed_after=0.342739083985
yarn_ccd_geometry_error=0.000000232831
impact_time=0.380000025034 removed_advance=0.123999990523
response_count=1 normal_impulse=12.399999618530
final_separation=-0.000000003725 final_fruit_x=-0.023999996483
response_error=0.000001202958 friction_contacts=1
cone_ratio=0.128008306026 slip_before=2.000001854884
slip_after=0.000000017553
self_ccd_position_error=0.000000357628
final_separation=0.000004277196 final_moving_height=-0.036001820117
max_correction=0.087999925017 present_contacts=0 swept_contacts=1
self_friction_velocity_error=0.000027455503 friction_contacts=1
cone_ratio=0.668487787247 slip_before=2.000063489379
slip_after=0.000050152179
energy_before=157.924313130071 energy_after=155.924161600600
momentum_error=0.000000478928
yarn_aerodynamics_velocity_error=0.000000930480
force=0.172502279282 force_error=0.000000008350
energy_before=25.000000000000 energy_after=24.993015079104
refinement_error=0.000000476837 co_moving_delta=0.000002861023
fruit_aerodynamics_velocity_error=0.000029432718
force=0.132944747806 force_error=0.000000004606
torque=0.000002209812 torque_error=0.000000000000
energy_before=3.155968230148 energy_after=3.148686517625
mouth_probe_inside_mask=0 released_mask=1 outside_mask=0 grazing_mask=0
edge_clearance_mask=1 edge_exit_mask=1 rotated_mask=1
clearance=0.025999993086 rotated_clearance=0.025999993086
fp64_qualified=true
state_hash=0x2ba7eb3acee2ad58
result=PASS
```

The eight normal-response-count differences, two fruit/yarn friction-count
differences, and one yarn/yarn friction-count difference are FP32/FP64
classifications of
already-resolved contacts whose separation is within the declared two-micron
surface tolerance. They are not hidden: the gate separately requires the
qualified control state, final penetration, accumulated impulse error, and
byte-identical Metal replay. Response-count equality itself is reported but is
not an acceptance condition.

GPU time is reported by the executable but is not yet a performance claim.
Qualification runs of this transaction ranged from `0.648646624992 s` to
`1.587818250002 s` under changing GPU contention, and the transaction has no
complete-trajectory baseline or profiler trace. Compact segment-pair records,
the triangular lookup, zeroed device buffers without host mirrors, and scoped
Metal autorelease pools reduced the executable's measured maximum resident set
from `401,801,216` to `235,192,320` bytes before the self-friction response
log; the qualified ABI-8 transaction measured `238,157,824` bytes, the ABI-9
air-load transaction measured `240,222,208` bytes, and the ABI-10
release-classification harness, including seven additional focused Metal/FP64
cases, measured `244,121,600` bytes. The memory-layout comparison itself
preserved its exact state hash and qualification values; later qualified
physics transactions intentionally changed the state hash. This is a
resource-footprint result, not a speed claim. The complete 16-test
Metal-labelled suite subsequently passed under the same shared-machine load.
The qualified combined metallib SHA-256 is
`bb3e0557983668fd0f69fae05c2ac0f2c4b92daa5255327078fd38844e02f3c5`.

## Remaining boundary

This result proves Metal ownership, FP64 equation agreement, deterministic
replay, and full-topology execution for free motion, axial yarn, strain
limiting, crossing-angle knots, yarn bending, ground-aware bend response, and
the ten-knot seam attachment, fruit free translation, fruit-pair normal
contact, cloth/fruit ground projection, full fruit/yarn present and swept
candidate geometry and normal response, full-topology swept/current yarn
self-contact with deterministic active-batch compaction, five contact-friction
families, fruit rolling resistance, and normalized fruit orientation
integration, full-yarn cylinder air loads, and fruit translational/rotational
air loads, plus topology-aware mouth release classification. It does
**not** yet
include:

- the complete grounded, spin, or pickup outcome on Metal.

Until those transactions reach Metal and match the FP64 replay, the README GIF
and its spill certificate remain CPU-reference evidence. Specimen calibration
is a separate physical-evidence requirement.
