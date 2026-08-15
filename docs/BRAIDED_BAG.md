# Braided-bag physics benchmark

## Purpose and model boundary

This benchmark is an executable GPU trajectory, not a collection of isolated
solver inputs. Six spherical particles fall under gravity inside a deformable
braided lattice for 480 steps at `dt = 1/480 s`. The complete loop—free motion,
contact generation, Delassus assembly, Temporal Cone solve, impulse
publication, and symplectic position advance—runs in one Metal command encoder
without per-step CPU readback.

The bag has 56 point-mass nodes in seven rings and 100 axial spring-damper
edges: two counter-wound helical families between adjacent rings and four
crossed bottom diameters. The eight mouth nodes are anchored. This is a
deliberately small braid mechanics model, not a shell, yarn-bending, or
self-contact model. Balls have translational mass and radius but no rotational
state. Each ball uses its nearest braid edge for wall contact; all 15 ball
pairs are present; and six contacts against a plane at `z=0` represent the
sub-ball-scale woven base. The plane is an explicit homogenized support
approximation, not hidden geometric evidence.

## Braid mechanics

For edge `(i,j)` with rest length `l_0`, current length `l`, direction `e`,
stiffness `k`, and axial damping `c`, the force accumulated by node `i` is

```math
f_{ij}=\left[k(l-l_0)+c(v_j-v_i)^T e\right]e.
```

Free velocity uses gravity and linear drag,

```math
v_i^*=v_i+\Delta t\left(g+m_i^{-1}\sum_j f_{ij}-d_i v_i\right).
```

Accepted contact impulses update velocity before the symplectic position
advance `x^{k+1}=x^k+dt v^{k+1}`. If the solver rejects a step, positions,
velocities, and warm starts are not partially published.

## Contact equations

Each contact stores up to three particles and scalar interpolation/sign
weights `w_ip`. For contact direction `d_ia` (`a` is normal, tangent-u, or
tangent-v), the Jacobian action is

```math
(Jv)_{i,a}=d_{i,a}^T\sum_p w_{i,p}v_p.
```

The GPU constructs every 3x3 Delassus block directly from shared particles:

```math
W_{(i,a),(j,b)}=
\sum_{p\in i\cap j}w_{i,p}w_{j,p}m_p^{-1}d_{i,a}^Td_{j,b}
+\delta_{ij}\delta_{ab}\,\mathrm{CFM}.
```

This makes the coupling physical and exact for the authored point-mass model:
wall contacts share braid nodes, ball contacts share balls, and unrelated
contacts have exact zero blocks. There are 27 deterministic candidates, but
the GPU admits only contacts whose current/predicted separation is at most
`0.01 m`. A SIMD prefix sum compacts them in canonical candidate order, then
builds exact sorted block CSR from the compacted particle participants. Warm
starts follow a stable physical identity rather than the compacted slot. An
empty active set is a valid, exactly certified solve. The retained capacities
are 27 contacts and 309 blocks; the qualified trajectory used at most 21
contacts and 181 blocks instead of 729 dense blocks.

Temporal Cone solves

```math
\min_{\lambda\in\mathcal C}
\frac12\lambda^T W\lambda+v_\mathrm{free}^T\lambda,
```

with one isotropic Coulomb cone per contact,

```math
\lambda_n\ge0,
\qquad
\sqrt{\lambda_u^2+\lambda_v^2}\le\mu\lambda_n.
```

Contact normals and both tangents are regenerated deterministically. Warm
starts are retained only when the stable contact identity matches.

## Independent success certificates

The trajectory does not accept `solver returned success` as sufficient. It
records and gates:

- normalized KKT gradient-mapping residual;
- cone feasibility in impulse units;
- normalized variational-inequality complementarity residual;
- nonpositive contact objective;
- maximum penetration and relative strand stretch;
- solver failures and transactional completion;
- ball escape through the wall, base, or mouth;
- byte-identical replay and dense/streamed trajectory equality.

Because the assembled operator is

```math
W=JM^{-1}J^T+\mathrm{CFM}\,I,
```

its positive-semidefinite response term gives

```math
\lambda_{\min}(W)\ge\mathrm{CFM},\qquad
\lambda_{\max}(W)\le\|W\|_\infty,
```

and therefore the runtime records and gates the rigorous upper bound

```math
\kappa_2(W)\le\frac{\|W\|_\infty}{\mathrm{CFM}}.
```

This is a bound, not an estimated condition number. Nonfinite values or a
preset-specific bound violation reject the run.

For residual `r=W lambda+v_free`, an uncapped elliptic cone additionally
requires dual feasibility and zero work:

```math
r_n\ge\sqrt{(\mu_u r_u)^2+(\mu_v r_v)^2},
\qquad
\lambda^T r=0.
```

The reported complementarity value is the maximum dual violation or the
absolute work gap divided by a bounded impulse-support scale. It is checked in
the iteration stop and again in final admission. This independent certificate
exposed a degenerate-cone defect: an inactive tangent component incorrectly
forced an otherwise interior one-axis `(normal, tangent)` pair to the wedge
boundary. The corrected projection drops the inactive component first and
projects only the active 2D wedge.

## Apple M4 matched result

Measured on 2026-08-16 with three replays per path:

```sh
./build/numi-solver-braided-bag \
  --environments 128 --steps 480 --replays 3
```

The one-second, 128-environment run passed with:

```text
failed_steps=0
escaped_mask=0
min_active_contacts=0
max_active_contacts=21
average_active_contacts=12.405257161
max_active_blocks=181
max_penetration=0.002769157
max_relative_stretch=0.136324883
max_kkt_residual=0.000001998
max_cone_violation=0.000000030
max_complementarity_residual=0.000002000
max_positive_objective=0.000000000
max_operator_infinity_norm=121.167037964
max_condition_upper=2423.340820312
max_iterations=493
dense_deterministic=true
streamed_deterministic=true
dense_stream_bitwise=true
state_hash=0x205fd70f99aa7f64
dense_gpu_seconds=0.895662792
streamed_gpu_seconds=0.515733347
dense_to_stream_speedup=1.736678065
```

The allocated streamed operator remains fixed-capacity and used 33.8% of the
dense operator bytes. Computation visits only the dynamically emitted blocks;
the peak was 181 of the 309-block capacity.

### Scaling

All rows use 480 steps and three replay averages on the same Apple M4 build.
Every row had zero failures/escapes, deterministic replays, and byte-identical
dense/streamed states and certificates.

| Environments | Dense s | Streamed s | Speedup | Streamed env-microsteps/s |
|---:|---:|---:|---:|---:|
| 16 | 0.619194222 | 0.366568153 | 1.689x | 20,951 |
| 32 | 0.493069292 | 0.336931917 | 1.463x | 45,588 |
| 64 | 0.618551097 | 0.376852750 | 1.641x | 81,517 |
| 128 | 0.895662792 | 0.515733347 | 1.737x | 119,131 |
| 256 | 1.757964042 | 0.940558750 | 1.869x | 130,646 |

An environment-microstep means one completed physics step for one bag; it is
not an RL transition. These command-buffer GPU times include the full bag
microstep loop, not CPU oracle work.

### Stress and convergence matrix

The stress presets use four environments, 480 steps, and three replays. The
baseline row above uses 128 environments. `low-cfm` reduces CFM from `0.05` to
`0.02`; `stiff-braid` raises axial stiffness from `80` to `320`; and
`high-friction` raises Coulomb friction from `0.60` to `1.20`.

| Preset | Condition upper | Max KKT | Max VI | Max iterations | Speedup |
|---|---:|---:|---:|---:|---:|
| baseline | 2423.34 | 1.998e-6 | 2.000e-6 | 493 | 1.737x |
| stiff-braid | 2233.95 | 1.954e-6 | 1.982e-6 | 422 | 1.612x |
| low-cfm | 6007.80 | 1.998e-6 | 2.000e-6 | 680 | 1.410x |
| high-friction | 1979.88 | 2.000e-6 | 2.000e-6 | 440 | 1.436x |

Every stress row had zero failed steps and escapes and exact dense/streamed
parity. A four-second, 1,920-step run also passed with zero failures/escapes,
`0.002601266 m` maximum penetration, `0.177796379` maximum relative stretch,
`1.999e-6` maximum VI residual, condition upper bound `2460.12`, and 543
maximum iterations. Halving the timestep from `1/480 s` to `1/960 s` for the
same simulated second produced a deterministic final-position L-infinity
difference of `0.025897510 m`, below the declared `0.075 m` refinement gate.

This proves the streamed Numi block-CSR path is superior to the dense Numi path
for this exact operator, trajectory, device, and build: identical FP32 states
and certificates, 33.8% of allocated operator memory, less operator work after
active compaction, and a measured 1.41–1.87x speedup across the qualified
matrix. It does not establish superiority over MuJoCo, Isaac Lab, or another
external solver because no matched external implementation was measured.

Metal dispatch-boundary counter sampling is unsupported on this device/toolchain,
so no stage-level occupancy, bandwidth, or counter percentage is claimed.

The qualified combined metallib SHA-256 is
`7dcbeb2528af383aedcbfbfda6e6836ad0b132cf4a15dc27a0524cdddbc2eee6`.

Use `--dump-obj PATH` to export the final braid lines and ball-center comments
for inspection. The OBJ is a geometry aid; the numerical and containment
certificates remain the proof surface.
