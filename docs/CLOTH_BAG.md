# Dense cloth produce-bag reference

## Purpose and evidence boundary

`numi-solver-cloth-bag` is a deterministic FP64 mechanics reference for an
open cotton produce bag containing twelve dynamic spheres. The live state is a
48 by 28 periodic wall joined to a structured 13 by 13 bottom: 1,465 particles,
2,880 triangles, 5,784 stretch/shear constraints, and 4,296 interior-edge bend
constraints.

This reference runs on the CPU. It does **not** prove a Metal cloth path,
Temporal Cone cloth contact, continuous collision detection, yarn-scale
friction, or hardware performance. It locks the cloth equations, topology,
contact geometry, three useful load cases, deterministic replay, and mesh export
before a GPU implementation is admitted.

## Cloth discretization

The grounded bag has a broad capped base, a scalloped ground skirt, a short
corrugated body, an inward folded irregular cuff, and no anchors. The top two
rows carry extra mass and a stiffer circumferential hem, but remain free. Warp,
weft, and both diagonal shear families use XPBD distance constraints.

The bottom uses a 13 by 13 square weave mapped continuously into the 48-node
circular wall boundary. Its 121 interior particles remove the former
single-center fan and its long, radially convergent constraints. Standard cloth
nodes carry `0.05 g` and hem nodes carry `0.10 g`, for a total authored bag mass
of `0.07805 kg` against `2.33 kg` of fruit. These are plausible authored values,
not calibrated measurements from a physical specimen.

For a constrained pair `(i,j)`,

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
escape. A fruit is classified as released once its center moves more than
`0.45 m` from the bag's bottom-center particle.

## Grounded pickup and pour

`--scenario pickup` starts from the grounded geometry and retains its collision
plane. It makes the same one rim particle kinematic, first lifting it and then
sweeping it sideways so gravity and the open mouth—not an authored fruit
animation—determine the spill. With
`s(x)=clamp(x,0,1)^2(3-2clamp(x,0,1))`,

```math
u(t)=s(t/0.58),\qquad p(t)=s((t-0.48)/0.62),
```

```math
g(t)=g_0+
\begin{bmatrix}
-0.54p & 0.12\sin(\pi p) &
0.72u+0.08\sin(\pi p)-0.12p
\end{bmatrix}.
```

Every other degree of freedom remains dynamic. Ground contact continues during
the whole motion, so released fruit falls to and settles on the plane.

## Contact

Sphere contact is generated against the actual cloth triangles, not a support
plane. For each sphere/triangle pair the reference first rejects a padded AABB,
then computes the exact closest point and barycentric weights. If distance `d`
is smaller than sphere radius plus cloth radius, an inequality projection
distributes the correction to all three cloth vertices and the sphere:

```math
C_c=d-(r_b+r_c)\ge0.
```

The triangle response caps each vertex's local contact inverse mass at the
inverse total cloth mass. This approximates the effective response of the
connected, tensioned patch instead of treating three `0.05 g` vertices as
isolated masses. It is a bounded CPU-reference approximation, not a substitute
for the future full coupled cloth response operator.

All 66 fruit pairs use nonpenetration constraints. Touching pairs receive a
deterministic, timestep-integrated tangential damping rate so a produce pile
has friction
instead of behaving like perfectly smooth marbles. The grounded and pickup
scenarios also project cloth and fruit against the plane. A deterministic
spatial hash enforces vertex-level cloth self-separation for nonlocal topology
pairs while
excluding the local two-ring neighborhood. Self-contact incidence is execution
evidence, not evidence of continuous triangle/triangle self-collision.

## Qualification gates

A run fails on nonfinite state, numerical escape, triangle collapse, excessive
warp/weft/shear/bottom strain, fruit/cloth penetration above `0.01 m`, vertex
self-penetration at or above the cloth diameter, excessive speed, or a
non-bit-identical replay. Grounded runs additionally require fruit containment
with `spilled_mask=0` and plane correction below one cloth radius. Extension
and compression are reported separately: the gate limits warp/weft/bottom
extension to `0.30` while
allowing up to `0.60` compression so a folded sheet is not rejected merely for
bunching.

Dihedral excursion is reported but intentionally is not capped as a failure:
large reversible folds are required textile behavior. Stability is instead
guarded by finite state, positive triangle area, bounded strain, contact
penetration, self-separation, and deterministic replay.

Run all three load cases:

```sh
./build/numi-solver-cloth-bag \
  --scenario grounded --steps 120 --substeps 4 --iterations 12 \
  --dump-obj grounded-1s.obj

./build/numi-solver-cloth-bag \
  --scenario spin --steps 60 --substeps 4 --iterations 12 \
  --dump-frames spin

./build/numi-solver-cloth-bag \
  --scenario pickup --steps 144 --substeps 4 --iterations 12 \
  --dump-frames pickup --dump-every 6
```

The executed 1.0-second grounded checkpoint on 2026-08-20 was deterministic
and passed with zero numerical escapes or physical spills:

```text
nodes=1465
triangles=2880
stretch_constraints=5784
bend_constraints=4296
balls=12
cloth_mass_kg=0.078050000
fruit_mass_kg=2.330000000
max_warp_strain=0.221845069
max_warp_extension=0.076170409
max_warp_compression=0.221845069
max_weft_strain=0.011401548
max_shear_strain=0.026139980
max_bottom_extension=0.006337407
max_bend_error=0.226902558
min_triangle_area=0.000056068
max_ball_penetration=0.004879665
max_self_penetration=0.002980520
ball_triangle_contacts=75624
self_contacts=17280
escaped_mask=0
spilled_mask=0
max_ground_penetration=0.001865301
deterministic=true
state_hash=0xfe292cef5e3eb5d4
result=PASS
```

The matching 0.5-second one-point spin checkpoint also passed. It reached
`0.098670795` maximum warp extension and `0.056124986` maximum shear strain,
recorded `29,912` fruit/triangle contacts, and replayed bit-identically with
`state_hash=0x3108f4c7e1ce7a76`. `released_mask=3584` records fruits 9, 10,
and 11 leaving the open mouth; `escaped_mask=0` distinguishes that physical
release from numerical divergence.

The 1.2-second grounded pickup checkpoint passed with
`max_warp_extension=0.194424871`, `max_shear_strain=0.095378970`,
`max_bottom_extension=0.007369176`, `max_ball_penetration=0.004879665`,
`max_self_penetration=0.004137716`, and `escaped_mask=0`. Its
`released_mask=3745` records fruits 0, 5, 7, 9, 10, and 11 leaving the bag;
four are on the plane at the final checkpoint and two remain in flight. The
independent replay matched `state_hash=0xbf062725eeee95dc`
bit-for-bit.

`--dump-obj` writes the actual simulated vertices and triangles plus fruit
center/radius/appearance comments. The spin and pickup scenarios also record
the exact kinematic grip position. `--dump-frames PREFIX` captures the initial,
quarter, half, three-quarter, and final states from the first authoritative
trajectory. Adding `--dump-every N` captures every Nth step plus the final
state. The second trajectory independently verifies the final hash. A rendered
replay is visual evidence for this FP64 trajectory only; it is not evidence for
the future Metal path. `tools/render_cloth_obj.swift` rasterizes exported state
with AppKit and does not advance or pose the simulation;
`tools/compose_cloth_gif.swift` combines those PNGs into a looping GIF without
interpolating new physics states.

The rasterizer draws the mapped row and column yarn families across the same
13 by 13 bottom surface used by contact. The visual bottom is therefore closed
without inventing the former radial fan.

See [CLOTH_REALISM_AUDIT.md](CLOTH_REALISM_AUDIT.md) for the remaining limits;
this checkpoint does not claim yarn-scale contact, calibrated textile material,
continuous collision, or a Metal cloth runtime.
