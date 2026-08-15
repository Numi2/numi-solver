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
contacts have exact zero blocks. The fixed 27-contact problem has 309 sorted
3x3 blocks instead of 729 dense blocks.

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

Measured on 2026-08-16:

```sh
./build/numi-solver-braided-bag \
  --environments 128 --steps 480 --replays 2
```

Each of three independent invocations passed with the same final state hash
`0x8c77acd537089a60`:

```text
failed_steps=0
escaped_mask=0
max_penetration=0.002773538
max_relative_stretch=0.135187015
max_kkt_residual=0.000001095
max_cone_violation=0.000000030
max_complementarity_residual=0.000001095
max_positive_objective=0.000000000
max_iterations=428
dense_deterministic=true
streamed_deterministic=true
dense_stream_bitwise=true
dense_operator_bytes=4718592
streamed_operator_bytes=1596416
stream_to_dense_operator_memory=0.338324653
```

The three paired GPU measurements were:

| Run | Dense seconds | Streamed seconds | Dense/stream speedup |
|---:|---:|---:|---:|
| 1 | 0.888583479 | 0.444437354 | 1.999344724x |
| 2 | 0.886406542 | 0.456465333 | 1.941892356x |
| 3 | 0.954061042 | 0.521337750 | 1.830024857x |

The median invocation delivered 134,599 streamed environment-microsteps/s
versus 69,144 dense environment-microsteps/s. An environment-microstep here
means one completed physics step for one bag; it is not an RL transition.

This proves the streamed Numi block-CSR path is superior to the dense Numi path
for this exact operator, trajectory, device, and build: identical FP32 states
and certificates, 33.8% of the operator memory, and a measured 1.83–2.00x
speedup. It does not establish superiority over MuJoCo, Isaac Lab, or any
external solver because no matched external implementation was measured.

The qualified combined metallib SHA-256 is
`c06b3631bcaeb9906ef323cb9bf0f044722507c5102890ca347e81bbbfd26313`.

Use `--dump-obj PATH` to export the final braid lines and ball-center comments
for inspection. The OBJ is a geometry aid; the numerical and containment
certificates remain the proof surface.
