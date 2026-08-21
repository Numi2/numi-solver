# Explicit-yarn produce-bag reference

## Purpose and evidence boundary

`numi-solver-cloth-bag` is a deterministic FP64 mechanics reference for an
open cotton produce bag containing twelve dynamic spheres. The bottom is
physically closed; only the folded top mouth is open. The live model contains:

- 1,465 massive knot particles;
- 2,904 axial warp, weft, and bottom yarn segments;
- 1,369 finite-compliance crossing-angle knot constraints;
- 2,834 three-knot yarn-bend constraints; and
- twelve fruits with translation, solid-sphere rotation, and friction.

The 2,880 triangles in an exported OBJ exist only for rasterization and a
minimum-area topology certificate. They are not stretch, bend, fruit-contact,
or self-contact primitives.

This is a CPU reference. It does **not** prove a Metal cloth implementation,
Temporal Cone cloth contact, hardware performance, or calibration to a
particular physical bag. It does lock an executable topology, equations,
collision geometry, three load cases, deterministic replay, and exported
states for later implementations to match.

## Topology and mass

The wall is a periodic 48 by 28 yarn grid. A 13 by 13 square bottom weave maps
its boundary one-to-one onto the lowest 48-node wall ring, leaving no geometric
hole or single-center fan. The upper rows form an irregular inward cuff and
remain free except for the small two-row patch held by the grab.

Ordinary knot particles carry `0.05 g`; the two hem rows carry `0.10 g`. Total
bag mass is `0.07805 kg`, against `2.33 kg` of fruit. These are authored
estimates, not specimen measurements.

Each axial yarn segment uses the standard XPBD distance equation

```math
C_s(x)=\|x_j-x_i\|-\ell_0,
\qquad
\alpha=\frac{c}{\Delta t^2},
```

```math
\Delta\lambda=
\frac{-C_s-\alpha\lambda}{w_i+w_j+\alpha}.
```

There are no diagonal sheet-shear springs. At a woven crossing, a compliant
knot instead preserves the authored angle between the local warp and weft
directions:

```math
C_k(x)=\hat e_w\cdot\hat e_f-
       \hat e_{w,0}\cdot\hat e_{f,0},
\qquad c_k=2\times10^{-6}.
```

Every three consecutive knots along a yarn receive a soft chord constraint,

```math
C_b(x)=\|x_{i+1}-x_{i-1}\|-\ell_{\mathrm{chord},0},
\qquad c_b=8\times10^{-2},
```

which resists yarn curvature without turning the net into a triangulated
sheet. A unilateral post-contact projection limits extension of every axial
yarn to `28.5%`; it does not resist compression or replace ordinary compliant
strain below that ceiling.

The two folded top-cuff rows use the same three-knot equation with authored
`c_b=1e-8`; the rest of the bag remains at `8e-2`. This represents the sewn,
doubled opening seam visible in the target bag rather than making all net yarn
rigid. The cuff value is not specimen-calibrated.

## Fruit, ground, and yarn contact

Fruit contact is sphere versus moving yarn capsule. For a sphere center `p`
and the closest point `q` on a yarn segment,

```math
C_{fy}=\|p-q\|-(r_f+r_y)\ge0,
\qquad r_y=0.004\ \mathrm{m}.
```

The correction uses the segment's actual closest-point weights and the true
inverse masses of its two endpoint knots. A piecewise active-set projection
stops either endpoint at its first ground-plane hit, activates unilateral
support, recomputes the response, and transfers only the remaining correction.
No artificial patch-mass cap is used.

Conservative advancement finds the first sphere/capsule hit using the relative
motion of the fruit and both yarn endpoints, so endpoint-separated states
cannot silently tunnel through one another. The independent
`--yarn-mechanics-probe` drives a sphere completely across a yarn in one step
and requires contact at the analytic combined radius. It also checks that a
60-degree knot crossing moves deterministically toward its 90-degree rest
angle under the production `2e-6` compliance, reducing angular error by more
than 25% without center drift.

Nonlocal self-contact is capsule versus capsule, with two-ring exclusions
derived from the axial yarn graph. Swept conservative advancement catches
crossing segments before endpoint resolution; local knot-particle collision
closes the remaining point cases. Render triangles never enter this path.
Completed published frames are certified below `2 um` for nonlocal yarn
overlap and extension-limit residual.

Fruit/yarn, fruit/fruit, fruit/plane, and yarn/yarn contacts apply a
maximum-dissipation tangent impulse capped by Coulomb friction:

```math
j_t=\min\left(\frac{\|v_t\|}{w_{\mathrm{eff}}},\mu j_n\right),
\qquad \|j_t\|\le\mu j_n.
```

The same endpoint weights used for normal contact distribute equal and
opposite friction impulses. Fruit contact-point velocity includes
`omega cross r`. Sorted primitive IDs keep the update order deterministic.

## Air and ground resistance

Air resistance is applied to the resolved yarn and fruit geometry instead of
using global velocity damping. For a yarn segment of length `L`, diameter `d`,
axis-parallel velocity `v_parallel`, and crossflow velocity `v_perp`,

```math
F_y=-\frac12\rho L\left[
C_d d\|v_\perp\|v_\perp+
C_f\pi d\|v_\parallel\|v_\parallel
\right].
```

The reference uses `rho=1.225 kg/m^3`, cylinder crossflow `C_d=1.10`, and
axial skin-friction `C_f=0.010`. Each segment force is evaluated from its
midpoint velocity and split equally between its endpoint masses. This makes
the total force invariant when a uniformly moving segment is subdivided.

Fruit use projected-area sphere drag and a surface-integrated rotational drag:

```math
F_f=-\frac12\rho C_{d,f}\pi r^2\|v\|v,
\qquad C_{d,f}=0.47,
```

```math
\tau_f=-\frac{3\pi^2}{8}\rho C_\omega r^5\|\omega\|\omega,
\qquad C_\omega=0.010.
```

Yarn forces are accumulated before a global dissipative attenuation; isolated
fruit translation and rotation use the exact time-integrated quadratic decay.
These updates cannot add relative kinetic energy or reverse relative velocity
in one substep. The run reports peak yarn force, peak fruit force and torque,
and the exact discrete relative kinetic energy removed by air loads.
`--aerodynamics-probe` checks analytic forces, energy reduction, spatial
subdivision invariance, exact isolated temporal refinement, a zero-force
co-moving-air case, and replay.

Cloth/plane motion uses load-dependent maximum-dissipation Coulomb friction,
not a fixed horizontal decay. The normal impulse is reconstructed from the
ground-active velocity response, and the tangent impulse satisfies

```math
j_t=\min(m\|v_t\|,\mu_g j_n),\qquad \mu_g=0.45.
```

Fruit rolling resistance is likewise load dependent. Its angular impulse is
capped by `mu_r r j_n`, with `mu_r=0.015`, and acts only on the two horizontal
rolling axes; it does not incorrectly damp vertical spin. Independent probes
cover sticking, cone-limited sliding, zero-load invariance, rolling-axis
stopping, vertical-spin preservation, and deterministic replay.

## Top-opening seam grab

The orange marker in the rendered replay is a virtual handle. It connects by
ten independent XPBD constraints to five neighboring knots on each of the two
folded opening-seam rows, each with `2e-4 m/N` compliance. No cloth particle is
pinned and the handle is not attached to the bottom. The visible orange
connector lines show the physical lag between the target and the ten massive
seam knots.

The airborne spin uses a smoothly accelerated orbit. The qualified grounded
pickup uses a lift, a brief loaded interval, a vertical downward snap, and a
slower recovery. With
`s(x)=clamp(x,0,1)^2(3-2clamp(x,0,1))`, its target is

```math
u(t)=s(t/0.80),\qquad d(t)=s((t-1.00)/0.23),
\qquad r(t)=s((t-1.45)/0.50),
```

```math
g(t)=g_0+
\begin{bmatrix}
-0.10d & 0.04\sin(\pi d) & 1.25u-0.85d+0.75r
\end{bmatrix}.
```

Only this seam target is authored. Every bag knot and fruit remains dynamic;
gravity, inertia, capsule contact, the open mouth, and the ground determine
the response. The downward cuff motion does not prescribe a fruit trajectory:
four fruits lag inertially, robustly exit the top opening, and remain spilled
on the ground. `released_mask` latches complete exit events and is kept
separate from `escaped_mask`, which denotes
numerical divergence.

Release is not inferred from distance to one bottom particle. Every substep
constructs an outward-oriented frame from all 48 upper-rim knots, projects the
ordered rim into that frame, and first records a sphere intersecting the mouth
region. Release requires the complete sphere to clear either through the cap or
around every 3D rim segment on the outward side by a further `25 mm`; this
hysteresis rejects grazing/re-entry chatter. `--mouth-release-probe` checks
contained, grazing, through-cap, edge-exit, far-outside, and rigidly rotated
opening cases with exact replay.

## Qualification

A qualified run uses two authoritative replays and requires their complete
state hashes to match. It fails on nonfinite state, numerical escape, collapsed
render topology, axial extension beyond the unilateral limit, excessive
fruit/yarn or yarn/yarn overlap, excessive speed, grip force above `500 N`, a
friction-cone violation, or a completed-frame certificate above `2 um`.
Grounded runs additionally require `spilled_mask=0` and bounded plane
correction. Pickup additionally requires at least two released fruits; the
qualified replay certifies four robust exits.

Run the three load cases and focused mechanics probes:

```sh
./build/numi-solver-cloth-bag \
  --scenario grounded --steps 120 --substeps 24 --iterations 32 --replays 2 \
  --dump-obj grounded-1s.obj

./build/numi-solver-cloth-bag \
  --scenario spin --steps 60 --substeps 48 --iterations 32 --replays 2 \
  --dump-frames spin

./build/numi-solver-cloth-bag \
  --scenario pickup --steps 240 --substeps 48 --iterations 32 --replays 2 \
  --dump-frames pickup --dump-every 10

./build/numi-solver-cloth-bag --rolling-probe
./build/numi-solver-cloth-bag --rolling-resistance-probe
./build/numi-solver-cloth-bag --mouth-release-probe
./build/numi-solver-cloth-bag --yarn-mechanics-probe
./build/numi-solver-cloth-bag --aerodynamics-probe
./build/numi-solver-cloth-bag --self-ccd-probe
./build/numi-solver-cloth-bag --strain-probe
./build/numi-solver-cloth-bag --self-friction-probe
./build/numi-solver-cloth-bag --cloth-ground-friction-probe
./build/numi-solver-cloth-bag --deformable-response-probe
```

The qualified 2026-08-21 checkpoints are:

| Scenario | Physical outcome | Published certificate | State hash |
|---|---|---|---|
| Grounded, `1.0 s` | no spill or escape | fruit/yarn `0.001 um`; final yarn overlap `0.934 um`; strain residual `0` | `0x58bbb2338d1a5369` |
| Spin, `0.5 s`, 48 substeps | visible lag/folding; no robust release or escape | fruit/yarn `0`; published yarn overlap `0.990 um`; strain residual `0` | `0xc0eb9fe5dd84d638` |
| Pickup, `2.0 s`, 48 substeps | four robust exits remain out; no escape | fruit/yarn `0.823 um`; published yarn overlap `0.937 um`; final overlap `0.002 um`; strain residual `0` | `0xe6aa91e33438537d` |

The pickup run reached `0.206217584` maximum axial warp extension,
`0.021853543` maximum bottom extension, `237.766842811 N` peak ten-knot grip
force, 103,969 swept fruit/yarn hits, 14,224 swept yarn/yarn hits, and 52,768
yarn/yarn friction contacts. `released_mask=3344`, `escaped_mask=0`, and the
friction-cone ratio never exceeded `1.0`.

Temporal refinement is an explicit boundary. At 48 substeps the two-replay
run above passes. At 96 substeps the same motion retains plural certified
release and every physical gate in the 1.5-second crossing window. At 24
substeps it produces only one release plus `2.160 mm` published fruit/yarn
overlap and `0.884 mm` published strain residual, so that resolution is
rejected. This is outcome-class refinement evidence, not equality of every
instantaneous peak or contact count across discretizations.

The direct deformable-response probe independently certifies true endpoint-mass
coupling: a free yarn and sphere preserve center of mass during separation; a
plane-supported yarn transfers the full `8 mm` to the sphere; and a yarn `1 mm`
above the plane first lands and then transfers the remaining `7 mm`. The
self-contact probe moves one yarn from `+0.02 m` to `-0.02 m` through another
in one step and stops at the `0.008 m` two-capsule thickness. The friction
probe certifies exact sticking, cone-limited sliding, momentum conservation,
dissipation, and deterministic replay.

## Export and visual evidence

`--dump-obj` writes the actual simulated knot positions plus render triangles,
fruit center/radius/orientation/angular velocity, and the exact virtual grip
target. `--dump-frames PREFIX` captures five evenly spaced states;
`--dump-every N` captures every Nth step plus the final state. The AppKit
renderer draws the axial yarn graph as cotton-like cylinders and the ten seam
connectors, but never modifies state. A PNG or GIF therefore visualizes the
qualified CPU trajectory; it is not independent proof of material calibration
or of a future Metal implementation.
