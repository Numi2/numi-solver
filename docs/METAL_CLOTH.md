# Metal cloth, seam grip, fruit/yarn, and yarn self-contact

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
- the unilateral 28.5% extension ceiling; and
- velocity publication from accepted positions.

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

For overlapping fruits `i,j`, the positional normal solve is

```math
\lambda=\frac{r_i+r_j-\|x_j-x_i\|}{w_i+w_j},\qquad
x_i\mathrel{-}=w_i\lambda n,\quad
x_j\mathrel{+}=w_j\lambda n.
```

The accumulated normal impulse is `lambda / dt`. Ground projection clamps
cloth to the yarn radius and fruit centers to their sphere radius, recording
the removed fruit advance as normal impulse. These are the CPU reference's
normal position transactions; tangential and rolling impulses remain a later
stage.

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
response as the CPU reference. Normal impulse is `lambda/dt`; tangential and
rolling response remain separate, later transactions.

## Independent qualification

The harness reconstructs the reference bag's 48 by 28 wall and 13 by 13
closed bottom, verifies the exact particle, distance, knot, bend, grip, fruit,
pair, and 34,848 fruit/yarn candidate counts, and rejects any graph color that
shares written state. It also checks all 4,149,792 nonlocal self-pairs and
every self-contact batch for unique knot writers. It executes 32
internal-constraint/grip/contact
iterations and three
extension-limit sweeps on Metal, then compares every particle, fruit, contact,
and multiplier with an independent FP64 implementation of the same colored
schedule. Two GPU runs must be byte-identical.

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
one substep. Metal must stop it on the original side at the `8 mm` combined
capsule diameter and match FP64 positions and accepted swept-contact count.

Run it with:

```sh
./build/numi-solver-cloth-metal \
  --replays 2 --iterations 32 --strain-sweeps 3
```

The Apple M4 result measured on 2026-08-21 was:

```text
abi=6 particles=1465 distances=2904 grips=10 knots=1369 bends=2834
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
max_fruit_velocity_error=0.000440123335
max_fruit_pair_penetration=0.000000000000
yarn_identity_exact=true yarn_control_qualified=true
current_yarn_overlaps=0 swept_yarn_impacts=0
max_yarn_separation_error=0.000000563364
max_yarn_current_normal_error=0.000056231269
max_active_yarn_weight_error=0.000004120541
yarn_response_count_exact=false
accepted_yarn_responses=113 expected_yarn_responses=105
response_count_mismatches=8
max_yarn_normal_impulse=0.003789614420
max_yarn_response_error=0.000000055079
max_yarn_penetration=0.000000000000
present_self_contacts=122 expected_present_self_contacts=123
swept_self_contacts=0 expected_swept_self_contacts=0
max_self_correction=0.003799978178
max_self_correction_error=0.000000002372
final_self_penetration=0.000000986641
max_strain_violation=0.000000000000
max_displacement=0.006799975103
grip_force=153.876787384189
strain_probe_initial_violation=0.215000003576
strain_probe_final_violation=0.000000007451
compression_length_change=0.000000000000
strain_probe_com_error=0.000000014901
ground_bend_position_error=0.000000056670
supported_height=0.004000000190
free_endpoint_rise=0.009750500321
fruit_pair_probe_position_error=0.000000039736
separation=2.000000044703 center_error=0.000000009934
normal_impulse=16.666667938232
ground_contact_position_error=0.000000000000
ground_contact_impulse_error=0.000000558794
cloth_height=0.004000000190 fruit_height=1.000000000000
fruit_normal_impulse=25.000000000000
yarn_ccd_geometry_error=0.000000059605
impact_time=0.380000025034 removed_advance=0.123999990523
response_count=1 normal_impulse=12.399999618530
final_separation=-0.000000003725 final_fruit_x=-0.023999996483
response_error=0.000000112880
self_ccd_position_error=0.000000000288
final_separation=0.000000001311 final_moving_height=0.008000001311
max_correction=0.087999999523 present_contacts=0 swept_contacts=1
state_hash=0xecae9af88f4eb569
result=PASS
```

The eight response-count differences are FP32/FP64 classifications of
already-resolved contacts whose separation is within the declared two-micron
surface tolerance. They are not hidden: the gate separately requires the
qualified control state, final penetration, accumulated impulse error, and
byte-identical Metal replay. Response-count equality itself is reported but is
not an acceptance condition.

GPU time is reported by the executable but is not yet a performance claim. The
instrumented run above reported `0.162993624981 s`, but the transaction has no
complete-trajectory baseline or profiler trace. Compact segment-pair records,
the triangular lookup, zeroed device buffers without host mirrors, and scoped
Metal autorelease pools reduced the executable's measured maximum resident set
from `401,801,216` to `234,455,040` bytes while preserving the exact state hash
and qualification values. This is a resource-footprint result, not a speed
claim. The complete 16-test Metal-labelled suite subsequently passed under the
same shared-machine load.
The qualified combined metallib SHA-256 is
`8d41bc5556a755a22ad65f50c95f1a5208d18a09b911f3e1fab3839b6ec25ce6`.

## Remaining boundary

This result proves Metal ownership, FP64 equation agreement, deterministic
replay, and full-topology execution for free motion, axial yarn, strain
limiting, crossing-angle knots, yarn bending, ground-aware bend response, and
the ten-knot seam attachment, fruit free translation, fruit-pair normal
contact, cloth/fruit ground projection, full fruit/yarn present and swept
candidate geometry and normal response, and full-topology swept/current yarn
self-contact with deterministic active-batch compaction. It does **not** yet
include:

- contact/strain reconciliation after the final limiter;
- friction, rolling resistance, or aerodynamic loads;
- fruit rotational integration, orientation publication, or release masks; or
- the complete grounded, spin, or pickup outcome on Metal.

Until those transactions reach Metal and match the FP64 replay, the README GIF
and its spill certificate remain CPU-reference evidence. Specimen calibration
is a separate physical-evidence requirement.
