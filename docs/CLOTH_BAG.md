# Dense cloth produce-bag reference

## Purpose and evidence boundary

`numi-solver-cloth-bag` is a deterministic FP64 mechanics reference for an
open cotton produce bag containing twelve dynamic spheres. The live state is a
48 by 28 periodic wall joined to a structured 13 by 13 bottom: 1,465 particles,
2,880 triangles, 5,784 stretch/shear constraints, and 4,296 interior-edge bend
constraints.

This reference runs on the CPU. It does **not** prove a Metal cloth path,
Temporal Cone cloth contact, cloth/cloth tangential friction, yarn-scale
contact, or hardware performance. It locks the cloth equations, topology,
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

After contact, a unilateral strain projection caps warp, weft, and bottom
extension at `28.5%` and shear at `38%`. It does not resist compression or
ordinary strain below those limits. Alternating primitive-contact and strain
passes prevent self-separation from silently turning the textile into rubber.

## Compliant-patch airborne spin

`--scenario spin` raises the same bag into free space. A virtual handle follows
a smoothly accelerated circular path:

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
The handle attaches to five neighboring rim particles through isotropic XPBD
constraints with `2e-3 m/N` compliance. All cloth particles and all twelve
fruits remain dynamic. Gravity,
inertial lag, the soft bend law, cloth contact, and the open mouth therefore
decide the response. Fruit release is recorded as `released_mask`; it is an
expected physical outcome for an open bag and is distinct from numerical
escape. A fruit is classified as released once its center moves more than
`0.45 m` from the bag's bottom-center particle.

## Grounded pickup and pour

`--scenario pickup` starts from the grounded geometry and retains its collision
plane. It drives the same compliant five-node rim patch, first lifting it and
then sweeping it sideways so gravity and the open mouth—not an authored fruit
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
the whole motion, so released fruit falls to and settles on the plane. Peak
attachment force and impulse are measured from the accumulated XPBD
multipliers; force above `500 N` fails qualification.

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

All 66 fruit pairs use nonpenetration constraints. Fruits carry solid-sphere
inertia, angular velocity, and a normalized quaternion. At fruit/cloth,
fruit/fruit, and fruit/plane contacts, the relative contact-point velocity
includes `omega cross r`; a maximum-dissipation tangential impulse is capped
by `|j_t| <= mu j_n` and applied with equal and opposite linear and angular
response. Grounded fruit additionally use timestep-integrated rolling
resistance.

Conservative advancement finds first time of impact between each linearly
moving fruit sphere and linearly moving cloth triangle. It uses a relative
motion bound, so a step cannot skip a zero of the sphere-to-triangle distance
function. The `--ccd-probe` deliberately moves a small sphere entirely through
a triangle with both endpoints separated and requires contact at the analytic
sphere-plus-cloth height.

The grounded and pickup scenarios stop predicted cloth/plane and fruit/plane
crossings at the plane before iterative contact. The removed free-flight
advance is reported separately from any actual post-contact penetration.

Cloth self-contact derives two-ring exclusions from the actual constraint
graph, including the structured bottom. Spatial grids broadphase nonlocal
vertex/triangle and edge/edge pairs. Conservative advancement prevents both
primitive classes from tunneling during prediction; endpoint projections and
strain reconciliation finish each substep. Qualification measures the final
nonlocal primitive overlap directly and requires less than `2 um`. Normal
separation is implemented; cloth/cloth tangential friction remains outside
this checkpoint.

## Qualification gates

A run fails on nonfinite state, numerical escape, triangle collapse, excessive
warp/weft/shear/bottom strain, fruit/cloth penetration above `0.01 m`, vertex
self-penetration at or above the cloth diameter, excessive linear/angular
speed, grip force above `500 N`, a friction-cone violation, or a
non-bit-identical replay. Final primitive self-overlap and strain-limit
violation must each remain below `2 um`. Grounded runs additionally require fruit containment
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

./build/numi-solver-cloth-bag --rolling-probe

./build/numi-solver-cloth-bag --ccd-probe

./build/numi-solver-cloth-bag --self-ccd-probe

./build/numi-solver-cloth-bag --strain-probe
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
max_warp_strain=0.265150491
max_warp_extension=0.265150491
max_warp_compression=0.221823022
max_weft_strain=0.010682636
max_shear_strain=0.028191510
max_bottom_extension=0.006237591
max_bend_error=0.414845908
min_triangle_area=0.000056064
max_ball_penetration=0.004879649
max_self_penetration=0.002980520
ball_triangle_contacts=71130
swept_ball_triangle_contacts=3231
self_contacts=21702
vertex_triangle_self_contacts=1441
edge_edge_self_contacts=2884
swept_edge_edge_self_contacts=98
final_primitive_self_penetration=0.000000000
final_strain_limit_violation=0.000000000
escaped_mask=0
spilled_mask=0
max_ground_penetration=0.001850024
max_swept_ground_advance=0.000761389
max_swept_ball_advance=0.003056590
max_angular_speed=11.827532489
max_friction_cone_ratio=1.000000000
deterministic=true
state_hash=0x657436d6056a72b8
result=PASS
```

The matching 0.5-second compliant-patch spin checkpoint also passed. It reached
`0.262473536` maximum warp extension and `0.090438378` maximum shear strain,
recorded `27,682` fruit/triangle contacts and 739 swept first-impact contacts,
reached `18.653576450 rad/s`, measured `3.707047521 N` peak attachment force, and
replayed bit-identically with
`state_hash=0x5acc3af1fbf386c5`. `released_mask=1536` records fruits 9 and 10
leaving the open mouth; `escaped_mask=0` distinguishes that physical
release from numerical divergence.

The 1.2-second grounded pickup checkpoint passed with
`max_warp_extension=0.285000000`, `max_shear_strain=0.077412270`,
`max_bottom_extension=0.006225497`, `max_ball_penetration=0.004879649`,
`max_self_penetration=0.003234989`, and `escaped_mask=0`. Its
`released_mask=4032` records fruits 6 through 11 leaving the bag;
four are on the plane at the final checkpoint and two remain in flight. The
maximum post-contact plane penetration was `0.001741407 m`, the maximum
discarded swept plane advance was `0.004898232 m`, and 2,078 swept
sphere/triangle contacts prevented up to `0.003056407 m` of relative advance.
The run resolved 1,740 vertex/triangle and 3,461 edge/edge endpoint contacts
plus 64 swept edge/edge impacts, with zero final primitive overlap and zero
strain-limit residual. Peak fruit angular speed was `31.035801118 rad/s`, and
peak five-node grip force was `2.855439024 N`. The independent replay matched
`state_hash=0xcf98b9fc01a6c90a`
bit-for-bit.

The independent `--rolling-probe` begins a solid sphere sliding at `1 m/s` and
reaches the analytic no-slip state `v = r omega = 5/7 m/s`, with zero reported
contact slip, the expected `5/7` energy ratio, cone ratio `1.0`, and an exact
deterministic replay.

The independent `--ccd-probe` moves a `0.02 m` sphere from `+0.08 m` to
`-0.08 m` through a triangle in one `0.001 s` step. Conservative advancement
stops it at `0.024000000560 m` against the analytic `0.024 m` target, removes
`0.104000000560 m` of tunneling advance, and replays exactly.

The independent `--self-ccd-probe` moves a vertex and a crossing edge from
`+0.02 m` to `-0.02 m` through another cloth primitive in one step. Both stop
at the `0.008 m` two-sided cloth thickness after removing `0.028 m` of
tunneling advance, and both replay exactly. `--strain-probe` independently
projects `50%` warp extension to exactly `28.5%` without center-of-mass drift.

`--dump-obj` writes the actual simulated vertices and triangles plus fruit
center/radius/appearance/orientation/angular-velocity comments. The spin and pickup scenarios also record
the exact virtual grip target. `--dump-frames PREFIX` captures the initial,
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
without inventing the former radial fan. It also draws a body-fixed fruit mark
from the exported quaternion so rolling and spin remain tied to solver state.

See [CLOTH_REALISM_AUDIT.md](CLOTH_REALISM_AUDIT.md) for the remaining limits;
this checkpoint does not claim yarn-scale contact, calibrated textile material,
cloth/cloth tangential friction, or a Metal cloth runtime.
