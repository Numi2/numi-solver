# Dense cloth-bag reference

## Purpose and evidence boundary

`numi-solver-cloth-bag` is a deterministic FP64 mechanics reference for a
closed fabric bag containing six dynamic spheres. It replaces the visual-only
idea of drawing a smooth skin over the 56-node braid with an executable cloth
state: 641 particles, 1,248 triangles, 2,496 stretch/shear constraints, and
1,856 interior-edge bending constraints.

This first reference runs on the CPU. It does **not** prove a Metal cloth path,
Temporal Cone cloth contact, continuous collision detection, yarn-scale
friction, or hardware performance. Its role is to lock the cloth equations,
topology, contact geometry, deterministic replay, and mesh export before a GPU
implementation is admitted.

## Cloth discretization

The surface is a periodic 32 by 20 triangulated grid plus a closed bottom cap.
The 32 mouth vertices are anchored. Warp, weft, and both diagonal shear
families use XPBD distance constraints. For a pair `(i,j)`,

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

Interior triangle pairs use a signed dihedral-angle XPBD constraint. The
gradient is independently evaluated by centered differences in FP64. This is
slower than an analytic production kernel but makes the reference small and
directly checkable.

```math
C_b(x)=\operatorname{wrap}(\theta(x)-\theta_0).
```

The authored bending compliance is much softer than stretch/shear compliance,
so the material can fold while retaining its woven lengths.

## Contact

Ball contact is generated against the actual cloth triangles, not a support
plane. For each sphere/triangle pair the reference computes the exact closest
point and barycentric weights. If distance `d` is smaller than sphere radius
plus cloth radius, an XPBD inequality projection distributes the correction to
all three cloth vertices and the dynamic sphere:

```math
C_c=d-(r_b+r_c)\ge0.
```

All 15 ball pairs are solved as nonpenetration constraints. A deterministic
spatial hash additionally enforces a vertex-level cloth self-separation
constraint for nonlocal topology pairs. The current trajectory does not
activate that self-contact path; this is measured zero incidence, not evidence
of continuous triangle/triangle self-collision.

## Qualification gates

A run fails on nonfinite state, any escaped sphere, any anchor drift, triangle
collapse, excessive warp/weft/shear strain, dihedral error above `0.5 rad`,
ball/cloth penetration above `0.01 m`, or a non-bit-identical replay.

The four-second qualification command is:

```sh
./build/numi-solver-cloth-bag \
  --steps 480 --substeps 2 --iterations 12 \
  --dump-obj cloth-4s.obj
```

Measured on 2026-08-16:

```text
nodes=641
triangles=1248
stretch_constraints=2496
bend_constraints=1856
balls=6
simulated_seconds=4.000000000
max_warp_strain=0.055238668
max_weft_strain=0.165018715
max_shear_strain=0.053532051
max_bend_error=0.285957975
min_triangle_area=0.000587580
max_ball_penetration=0.006600682
max_self_penetration=0.000000000
ball_triangle_contacts=175514
self_contacts=0
escaped_mask=0
max_anchor_error=0.000000000
deterministic=true
state_hash=0xce8f79793de6681b
result=PASS
```

`--dump-obj` writes the actual simulated vertices and triangles plus sphere
center/radius comments. A rendered surface is visual evidence for this FP64
reference trajectory only; it is not evidence for the future Metal path.
