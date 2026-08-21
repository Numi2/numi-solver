# Metal cloth internal-constraint, fruit, and seam-grip transaction

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

## Independent qualification

The harness reconstructs the reference bag's 48 by 28 wall and 13 by 13
closed bottom, verifies the exact particle, distance, knot, bend, grip, fruit,
and pair counts, and rejects any graph color that shares written state. It
executes 32 internal-constraint/grip/contact iterations and three
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
FP64 normal impulse.

Run it with:

```sh
./build/numi-solver-cloth-metal \
  --replays 2 --iterations 32 --strain-sweeps 3
```

The Apple M4 result measured on 2026-08-21 was:

```text
abi=3 particles=1465 distances=2904 grips=10 knots=1369 bends=2834
fruits=12 fruit_pairs=66
distance_colors=5 knot_colors=6 bend_colors=5 fruit_pair_colors=15
failure_flags=0 deterministic=true
max_position_error=0.000000339950
max_velocity_error=0.001958112178
max_distance_lambda_error=0.000000000020
max_grip_lambda_error=0.000000000008
max_knot_lambda_error=0.000000000001
max_bend_lambda_error=0.000000000010
max_fruit_position_error=0.000000002342
max_fruit_velocity_error=0.000013488655
max_fruit_pair_penetration=0.000000000000
max_strain_violation=0.000000000000
max_displacement=0.003793269600
grip_force=153.055611476897
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
state_hash=0x8ad67bf5714bd987
result=PASS
```

GPU time is reported by the executable but is not yet a performance claim:
this transaction has no complete-trajectory baseline and no profiler trace.
The qualified combined metallib SHA-256 is
`8eb95bfbfade065c4d255d4957ee4ed9b8e1348e7ce0a4a753ecff10282f2122`.

## Remaining boundary

This result proves Metal ownership, FP64 equation agreement, deterministic
replay, and full-topology execution for free motion, axial yarn, strain
limiting, crossing-angle knots, yarn bending, ground-aware bend response, and
the ten-knot seam attachment, fruit free translation, fruit-pair normal
contact, and cloth/fruit ground projection. It does **not** yet include:

- sphere/yarn or yarn/yarn contact;
- continuous collision detection or contact/strain reconciliation;
- friction, rolling resistance, or aerodynamic loads;
- fruit rotational integration, orientation publication, or release masks; or
- the complete grounded, spin, or pickup outcome on Metal.

Until those transactions reach Metal and match the FP64 replay, the README GIF
and its spill certificate remain CPU-reference evidence. Specimen calibration
is a separate physical-evidence requirement.
