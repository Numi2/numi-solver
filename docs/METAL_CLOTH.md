# Metal cloth axial and seam-grip transaction

## Qualified scope

`numi-solver-cloth-metal` is the first device-owned transaction for the
explicit-yarn produce-bag topology. It is not the complete bag trajectory. The
versioned ABI in `include/numi/cloth_bag_gpu.h` owns:

- 1,465 massive cloth particles;
- 2,904 axial warp, weft, and bottom constraints;
- the ten finite-compliance top-seam grips;
- gravity prediction and symplectic position advance;
- XPBD axial-yarn projection;
- the unilateral 28.5% extension ceiling; and
- velocity publication from accepted positions.

The distance graph is greedily colored once into five deterministic batches.
No two constraints in a batch write the same particle. Each batch therefore
runs in parallel without float atomics or a lane-zero serial loop, while
ordered dispatches preserve a deterministic Gauss-Seidel schedule across
colors. The ten grips touch ten distinct seam particles and also execute in
parallel.

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
closed bottom, verifies the exact particle, distance, and grip counts, and
rejects any graph color that shares a particle. It executes 32 axial/grip
iterations and three extension-limit sweeps on Metal, then compares every
particle and multiplier with an independent FP64 implementation of the same
colored schedule. Two GPU runs must be byte-identical.

The focused strain case starts one segment with `0.215 m` excess extension and
another in compression. One Metal sweep must remove the extension violation,
leave the compressed length unchanged, preserve the stretched pair's center
of mass, and match the FP64 oracle.

Run it with:

```sh
./build/numi-solver-cloth-metal \
  --replays 2 --iterations 32 --strain-sweeps 3
```

The Apple M4 result measured on 2026-08-21 was:

```text
particles=1465 distances=2904 grips=10 colors=5
failure_flags=0 deterministic=true
max_position_error=0.000000229983
max_velocity_error=0.001324699865
max_distance_lambda_error=0.000000000009
max_grip_lambda_error=0.000000000004
max_strain_violation=0.000000000000
max_displacement=0.003786783037
grip_force=143.304384925706
strain_probe_initial_violation=0.215000003576
strain_probe_final_violation=0.000000007451
compression_length_change=0.000000000000
strain_probe_com_error=0.000000014901
state_hash=0x1079b64ace940500
result=PASS
```

GPU time is reported by the executable but is not yet a performance claim:
this transaction has no complete-trajectory baseline and no profiler trace.

## Remaining boundary

This result proves Metal ownership, FP64 equation agreement, deterministic
replay, and full-topology execution for free motion, axial yarn, strain
limiting, and the ten-knot seam attachment. It does **not** yet include:

- crossing-angle knot constraints or three-knot bend constraints;
- sphere/yarn, sphere/sphere, plane, or yarn/yarn contact;
- continuous collision detection or contact/strain reconciliation;
- friction, rolling resistance, or aerodynamic loads;
- fruit translation, rotation, orientation publication, or release masks; or
- the complete grounded, spin, or pickup outcome on Metal.

Until those transactions reach Metal and match the FP64 replay, the README GIF
and its spill certificate remain CPU-reference evidence. Specimen calibration
is a separate physical-evidence requirement.
