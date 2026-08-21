# Metal cloth internal-constraint and seam-grip transaction

## Qualified scope

`numi-solver-cloth-metal` is the first device-owned transaction for the
explicit-yarn produce-bag topology. It is not the complete bag trajectory. The
versioned ABI in `include/numi/cloth_bag_gpu.h` owns:

- 1,465 massive cloth particles;
- 2,904 axial warp, weft, and bottom constraints;
- 1,369 compliant crossing-angle knot constraints;
- 2,834 three-knot yarn-bend constraints;
- the ten finite-compliance top-seam grips;
- gravity prediction and symplectic position advance;
- XPBD axial-yarn projection;
- the unilateral 28.5% extension ceiling; and
- velocity publication from accepted positions.

The distance, knot, and bend graphs are greedily colored once into five, six,
and five deterministic batches. No two constraints in a batch write the same
particle. Each batch therefore runs in parallel without float atomics or a
lane-zero serial loop, while ordered dispatches preserve a deterministic
Gauss-Seidel schedule across colors. The ten grips touch ten distinct seam
particles and also execute in parallel. Bend projection includes the CPU
reference's active-set response for a yarn endpoint supported by the ground.

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

## Independent qualification

The harness reconstructs the reference bag's 48 by 28 wall and 13 by 13
closed bottom, verifies the exact particle, distance, knot, bend, and grip
counts, and rejects any graph color that shares a written particle. It executes
32 internal-constraint/grip iterations and three extension-limit sweeps on
Metal, then compares every particle and multiplier with an independent FP64
implementation of the same colored schedule. Two GPU runs must be
byte-identical.

The focused strain case starts one segment with `0.215 m` excess extension and
another in compression. One Metal sweep must remove the extension violation,
leave the compressed length unchanged, preserve the stretched pair's center
of mass, and match the FP64 oracle. A separate ground-bend case requires a
supported endpoint to remain at the yarn radius while the free endpoint moves,
again matching an independent FP64 active-set solve.

Run it with:

```sh
./build/numi-solver-cloth-metal \
  --replays 2 --iterations 32 --strain-sweeps 3
```

The Apple M4 result measured on 2026-08-21 was:

```text
abi=2 particles=1465 distances=2904 grips=10 knots=1369 bends=2834
distance_colors=5 knot_colors=6 bend_colors=5
failure_flags=0 deterministic=true
max_position_error=0.000000339950
max_velocity_error=0.001958112178
max_distance_lambda_error=0.000000000020
max_grip_lambda_error=0.000000000008
max_knot_lambda_error=0.000000000001
max_bend_lambda_error=0.000000000010
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
state_hash=0x92c8f09cc705dab8
result=PASS
```

GPU time is reported by the executable but is not yet a performance claim:
this transaction has no complete-trajectory baseline and no profiler trace.
The qualified combined metallib SHA-256 is
`fadb4f7084c2c9fe468e2f382e922047db0c4744eaf6e2e377dd27c4603f726e`.

## Remaining boundary

This result proves Metal ownership, FP64 equation agreement, deterministic
replay, and full-topology execution for free motion, axial yarn, strain
limiting, crossing-angle knots, yarn bending, ground-aware bend response, and
the ten-knot seam attachment. It does **not** yet include:

- sphere/yarn, sphere/sphere, plane, or yarn/yarn contact;
- continuous collision detection or contact/strain reconciliation;
- friction, rolling resistance, or aerodynamic loads;
- fruit translation, rotation, orientation publication, or release masks; or
- the complete grounded, spin, or pickup outcome on Metal.

Until those transactions reach Metal and match the FP64 replay, the README GIF
and its spill certificate remain CPU-reference evidence. Specimen calibration
is a separate physical-evidence requirement.
