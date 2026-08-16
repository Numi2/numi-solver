# Dense cloth produce-bag reference

## Purpose and evidence boundary

`numi-solver-cloth-bag` is a deterministic FP64 mechanics reference for an
open cotton produce bag containing eight dynamic spheres. The live state is a
48 by 28 periodic cloth grid plus a bottom cap: 1,345 particles, 2,640
triangles, 5,280 stretch/shear constraints, and 3,936 interior-edge bend
constraints.

This reference runs on the CPU. It does **not** prove a Metal cloth path,
Temporal Cone cloth contact, continuous collision detection, yarn-scale
friction, or hardware performance. It locks the cloth equations, topology,
contact geometry, two useful load cases, deterministic replay, and mesh export
before a GPU implementation is admitted.

## Cloth discretization

The grounded bag has a narrow capped base, fruit-sized body, flared loose rim,
and no anchors. The top two rows carry extra mass and a stiffer circumferential
hem, but remain free. Warp, weft, and both diagonal shear families use XPBD
distance constraints. For a pair `(i,j)`,

```math
C(x)=\|x_j-x_i\|-\ell_0,
\qquad
\alpha=\frac{c}{\Delta t^2},
```

```math
\Delta\lambda=
\frac{-C-\alpha\lambda}
{w_i+w_j+\alpha},
\qquad
x_k\leftarrow x_k+w_k\nabla C_k\Delta\lambda.
```

Interior triangle pairs use a signed dihedral-angle XPBD constraint. Its
gradient is evaluated by centered differences in FP64. The bend compliance is
`8e-4`, deliberately much softer than stretch/shear, so the textile can sag,
fold, and twist without losing its woven lengths.

```math
C_b(x)=\operatorname{wrap}(\theta(x)-\theta_0).
```

## One-point airborne spin

`--scenario spin` raises the same bag into free space and makes exactly one rim
particle kinematic. The grip follows a smoothly accelerated circular path:

```math
\theta(t)=\omega\left[t-\tau\left(1-e^{-t/\tau}\right)\right],
```

```math
g(t)=g_0+
\begin{bmatrix}
R(\cos\theta-1) & R\sin\theta & a\sin(\theta/2)
\end{bmatrix},
```

with `R=0.28 m`, `omega=4.8 rad/s`, `tau=0.18 s`, and `a=0.035 m`.
Every other cloth particle and all eight fruits remain dynamic. Gravity,
inertial lag, the soft bend law, cloth contact, and the open mouth therefore
decide the response. Fruit release is recorded as `released_mask`; it is an
expected physical outcome for an open bag and is distinct from numerical
escape.

## Contact

Sphere contact is generated against the actual cloth triangles, not a support
plane. For each sphere/triangle pair the reference first rejects a padded AABB,
then computes the exact closest point and barycentric weights. If distance `d`
is smaller than sphere radius plus cloth radius, an inequality projection
distributes the correction to all three cloth vertices and the sphere:

```math
C_c=d-(r_b+r_c)\ge0.
```

All 28 fruit pairs use nonpenetration constraints. The grounded scenario also
projects cloth and fruit against the plane. A deterministic spatial hash
enforces vertex-level cloth self-separation for nonlocal topology pairs. Zero
self-contact incidence in a run is execution evidence, not evidence of
continuous triangle/triangle self-collision.

## Qualification gates

A run fails on nonfinite state, numerical escape, triangle collapse, excessive
warp/weft/shear strain, fruit/cloth penetration above `0.01 m`, vertex
self-penetration at or above the cloth diameter, excessive speed, or a
non-bit-identical replay. Grounded runs additionally require fruit containment
and bounded plane penetration.

Dihedral excursion is reported but intentionally is not capped as a failure:
large reversible folds are required textile behavior. Stability is instead
guarded by finite state, positive triangle area, bounded strain, contact
penetration, self-separation, and deterministic replay.

Run both load cases:

```sh
./build/numi-solver-cloth-bag \
  --scenario grounded --steps 30 --substeps 4 --iterations 12 \
  --dump-obj grounded-025s.obj

./build/numi-solver-cloth-bag \
  --scenario spin --steps 90 --substeps 4 --iterations 12 \
  --dump-obj spin-075s.obj
```

The measured 0.25-second spin checkpoint on 2026-08-16 was deterministic and
passed with zero numerical escapes:

```text
nodes=1345
triangles=2640
stretch_constraints=5280
bend_constraints=3936
balls=8
max_warp_strain=0.102616622
max_weft_strain=0.037973637
max_shear_strain=0.045706399
max_bend_error=0.442571213
min_triangle_area=0.000184935
max_ball_penetration=0.002873771
max_self_penetration=0.000000000
escaped_mask=0
max_speed=3.720392619
deterministic=true
state_hash=0xbb92c8b152247227
result=PASS
```

The 0.75-second continuation also passed. Peak warp/shear strain reached
`0.176413300` / `0.072014810`, fruit/cloth penetration remained
`0.002873771 m`, all 1,345 cloth vertices stayed finite, and replay remained
bit-identical (`state_hash=0x851e5ce550037b96`). `released_mask=224` records
that fruits 5, 6, and 7 left the deliberately open mouth; `escaped_mask=0`
confirms this was a bounded physical outcome rather than numerical divergence.

`--dump-obj` writes the actual simulated vertices and triangles plus fruit
center/radius/appearance comments. A rendered replay is visual evidence for
this FP64 trajectory only; it is not evidence for the future Metal path.
