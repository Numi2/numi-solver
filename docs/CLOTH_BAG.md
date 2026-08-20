# Dense cloth produce-bag reference

## Purpose and evidence boundary

`numi-solver-cloth-bag` is a deterministic FP64 mechanics reference for an
open cotton produce bag containing twelve dynamic spheres. The live state is a
48 by 28 periodic cloth grid plus a bottom cap: 1,345 particles, 2,640
triangles, 5,280 stretch/shear constraints, and 3,936 interior-edge bend
constraints.

This reference runs on the CPU. It does **not** prove a Metal cloth path,
Temporal Cone cloth contact, continuous collision detection, yarn-scale
friction, or hardware performance. It locks the cloth equations, topology,
contact geometry, two useful load cases, deterministic replay, and mesh export
before a GPU implementation is admitted.

## Cloth discretization

The grounded bag has a broad capped base, a short corrugated body, an inward
folded irregular cuff, and no anchors. The top two rows carry extra mass and a
stiffer circumferential hem, but remain free. Warp, weft, and both diagonal
shear families use XPBD distance constraints. For a pair `(i,j)`,

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
`8e-2`, deliberately much softer than stretch/shear, so the textile can sag,
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
Every other cloth particle and all twelve fruits remain dynamic. Gravity,
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

All 66 fruit pairs use nonpenetration constraints. Touching pairs receive a
deterministic tangential-velocity damping step so a produce pile has friction
instead of behaving like perfectly smooth marbles. The grounded scenario also
projects cloth and fruit against the plane. A deterministic spatial hash
enforces vertex-level cloth self-separation for nonlocal topology pairs while
excluding the local two-ring neighborhood. Self-contact incidence is execution
evidence, not evidence of continuous triangle/triangle self-collision.

## Qualification gates

A run fails on nonfinite state, numerical escape, triangle collapse, excessive
warp/weft/shear strain, fruit/cloth penetration above `0.01 m`, vertex
self-penetration at or above the cloth diameter, excessive speed, or a
non-bit-identical replay. Grounded runs additionally require fruit containment
with `spilled_mask=0` and bounded plane penetration. Extension and compression
are reported separately: the gate limits warp/weft extension to `0.30` while
allowing up to `0.60` compression so a folded sheet is not rejected merely for
bunching.

Dihedral excursion is reported but intentionally is not capped as a failure:
large reversible folds are required textile behavior. Stability is instead
guarded by finite state, positive triangle area, bounded strain, contact
penetration, self-separation, and deterministic replay.

Run both load cases:

```sh
./build/numi-solver-cloth-bag \
  --scenario grounded --steps 120 --substeps 4 --iterations 12 \
  --dump-obj grounded-1s.obj

./build/numi-solver-cloth-bag \
  --scenario spin --steps 30 --substeps 4 --iterations 12 \
  --dump-obj spin-025s.obj
```

The measured 1.0-second grounded checkpoint on 2026-08-20 was deterministic
and passed with zero numerical escapes or physical spills:

```text
nodes=1345
triangles=2640
stretch_constraints=5280
bend_constraints=3936
balls=12
max_warp_strain=0.106840854
max_warp_extension=0.098259682
max_warp_compression=0.106840854
max_weft_strain=0.029464806
max_shear_strain=0.037335302
max_bend_error=0.412563172
min_triangle_area=0.000050047
max_ball_penetration=0.006000033
max_self_penetration=0.004564645
ball_triangle_contacts=76233
self_contacts=94078
escaped_mask=0
spilled_mask=0
max_ground_penetration=0.007905552
deterministic=true
state_hash=0xc404b0ad6d16b0db
result=PASS
```

The matching 0.25-second one-point spin checkpoint also passed. It kept all
twelve fruits contained, reached `0.160134927` maximum warp extension and
`0.056010842` maximum shear strain, recorded `5,065` fruit/triangle contacts,
and replayed bit-identically with `state_hash=0xa8b114cc13f79ce6`.

`--dump-obj` writes the actual simulated vertices and triangles plus fruit
center/radius/appearance comments. A rendered replay is visual evidence for
this FP64 trajectory only; it is not evidence for the future Metal path.
`tools/render_cloth_obj.swift` rasterizes that exported state with AppKit and
does not advance or pose the simulation.
