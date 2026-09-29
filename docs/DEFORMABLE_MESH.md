# Native shared-node elastic mesh drop

[![Native mesh compresses and recovers on the plane](assets/deformable-mesh-refined-drop.gif)](assets/deformable-mesh-refined-drop.mp4)

This Apple Metal trajectory drops a nonlinear elastic volume onto the support
plane. Its 309 shared nodes and 1,280 tetrahedra own one set of positions, velocities,
and lumped masses. The surface is the exact 320-triangle boundary of those
elements. The renderer preserves that faceted geometry without sphere fitting,
smoothing, posing, or animation forces. The video plays 101 native states at
30 fps: 3.367 seconds of playback represent 0.5 simulated seconds.

| Before contact | Maximum captured compression | Later recovery |
| --- | --- | --- |
| ![Native elastic mesh above the plane](assets/deformable-mesh-refined-drop-0.png) | ![Native elastic mesh compressed against the plane at 0.185 seconds](assets/deformable-mesh-refined-drop-37.png) | ![Native mesh recovering at 0.375 seconds](assets/deformable-mesh-refined-drop-75.png) |

## Mechanics and contact

Each element evaluates the [qualified neo-Hookean constitutive law](DEFORMABLE_VOLUMES.md)
on its four owning node indices. Deterministic, sorted incidence lists gather
all elastic contributions at each node; no floating-point atomics are used.
Density is authored as `1000 kg/m³`, and each tetrahedron distributes one
quarter of its mass to each corner. The resulting total mass is `0.317019 kg`.
The authored coefficients are `mu=3000 Pa` and `lambda=6000 Pa`; they are not
measured fruit properties.

The force-integrated nodal velocities advance positions under gravity. A
predicted plane impact publishes `z=0` and zero incoming normal velocity,
with an impulse computed from that node's own mass and velocity change.
Tangential velocity is retained. This benchmark uses a frictionless inelastic
plane; stored elastic energy can subsequently lift nodes and recover shape.
It does not use a prescribed squash, rebound, center path, or bulk fruit pose.

Every complete candidate mesh is evaluated before acceptance. One inverted
element, invalid mass, malformed incidence list, or ABI failure rejects the
whole step. All accepted positions, velocities, and the support ledger remain
unchanged on rejection. Failure flags persist across every step, including
steps between captured states. A separate metallib preserves the running
cloth replay and the earlier independent-tetrahedron qualification artifacts.

## Refinement with the same authored body

Uniform eight-child tetrahedron refinement adds shared edge midpoints. It
preserves the original piecewise-flat surface, material coefficients, density,
and all pre-existing rest nodes. It does not project new nodes to a sphere.
Lumped masses and deterministic element incidence are rebuilt from the refined
volumes. Total mass changes by less than 12 micrograms from FP32 rest-coordinate
rounding. Each refinement retains a closed, outward-oriented surface and
exactly two opposing owners for every internal element face.

| Resolution | Owning nodes / tetrahedra / surface triangles | Minimum height | Final height | Maximum half-step nodal difference |
| --- | --- | ---: | ---: | ---: |
| Original | 13 / 20 / 20 | 50.89 mm | 77.63 mm | 59.70 µm |
| One refinement | 55 / 160 / 80 | 49.14 mm | 102.66 mm | 105.98 µm |
| Two refinements, newest video | 309 / 1,280 / 320 | 47.85 mm | 79.60 mm | 135.85 µm |

All three actual 0.5-second native simulations pass two exact replays, the
5,000-versus-10,000-step comparison, five whole-mesh rejection checks, zero
plane penetration, positive accepted element volumes, and the existing
momentum/energy-increase bounds. The new default-resolution run preserves all
101 original OBJ snapshots byte for byte. The finest run reaches a minimum
all-step volume ratio of 0.27089 and an accumulated plane impulse of 1.78145 Ns.
Its maximum vertical momentum discrepancy is 1.359 µNs. These are authored
frictionless-plane benchmarks, not calibrated fruit specimens.

Spatial convergence is still unresolved. At matched pre-existing nodes and
equal exported times, the 20-to-160-element comparison differs by up to
37.18 mm; the 160-to-1,280-element comparison differs by 36.39 mm. The latter
exceeds the explicit 1 mm spatial benchmark target and its audit returns FAIL.
Rebound shape and phase depend substantially on discretization despite passing
individual timestep checks. The [comparison report](assets/deformable-mesh-refinement-audit.json)
retains those results. Increasing element count does not erase this failure.

[Finest qualification log](assets/deformable-mesh-refined-drop-qualified.log),
[source, binary, state and video fingerprints](assets/deformable-mesh-refined-drop-evidence.json),
[finest nodal trace](assets/deformable-mesh-refined-drop-trajectory.csv), and
[owning tetrahedra](assets/deformable-mesh-refined-drop-topology.csv).

```sh
./build/numi-solver-deformable-mesh --mesh-refinement 2 \
  --trajectory build/deformable-mesh-level-2
python3 tools/audit_deformable_mesh_trajectory.py build/deformable-mesh-level-2
# Produce all three levels before comparing their matched nodes.
python3 tools/audit_deformable_mesh_refinement.py \
  build/deformable-mesh-level-0 build/deformable-mesh-level-1 \
  build/deformable-mesh-level-2 --output build/deformable-mesh-refinement-audit.json
```

## Next native spatial refinement

A third uniform refinement adds a fourth resolution: **2,057 shared nodes,
10,240 tetrahedra and 1,280 boundary triangles**. It preserves the same
piecewise-flat body, material and pre-existing rest nodes. Construction now
builds incidence once per element and gathers by owning node; the deterministic
element/corner order remains byte-identical to the previous full scan at all
four resolutions. The [CPU topology probe](assets/deformable-mesh-topology-3-evidence.json)
passes without occupying the GPU used by the full cloth replay.

This is construction evidence. The new native resolution still requires its
complete 0.5-second run, two exact replays, half timestep, five transactional
rejections and independent topology/trajectory audit before publication as
qualified media. The existing 36.39 mm spatial comparison remains FAIL until
a new matched-node comparison establishes otherwise. Higher element count
alone is not convergence.

```sh
./build/numi-solver-deformable-mesh --topology-probe
./build/numi-solver-deformable-mesh --mesh-refinement 3 \
  --trajectory build/deformable-mesh-level-3
python3 tools/audit_deformable_mesh_trajectory.py build/deformable-mesh-level-3
```

## Original 20-element M4 qualification

The [original coarse video](assets/deformable-mesh-drop.mp4) remains available.


[Native qualification log](assets/deformable-mesh-drop-qualified.log),
[source, binary, state and video fingerprints](assets/deformable-mesh-drop-evidence.json),
[nodal trace](assets/deformable-mesh-drop-trajectory.csv), and
[owning tetrahedra](assets/deformable-mesh-drop-topology.csv).

| Check | Result |
| --- | ---: |
| Native simulation / half-step refinement | 5,000 / 10,000 steps over 0.5 s |
| Two complete native replays | Exact positions, velocities, and ledgers at all 101 captures |
| Invalid candidates in either completed run or refinement | Zero |
| Mesh height: initial / minimum / final | 85.07 / 50.89 / 77.63 mm |
| Maximum relative nodal shape change | 24.75 mm |
| Maximum nodal difference under timestep refinement | 59.70 µm |
| Minimum volume ratio across every coarse step and element | 0.59891 |
| Plane penetration | Zero |
| Maximum net internal force, coarse run | 1.314 µN |
| Accumulated plane normal impulse, coarse run | 1.56416 Ns |
| Maximum vertical momentum-balance discrepancy | 4.175 µNs |
| Maximum sampled mechanical-energy increase above the initial state | Zero |
| Maximum elastic energy | 0.31223 J |
| Maximum pre-contact center-of-mass free-fall error | 0.345 µm |
| Invalid mass, inversion, node index, incidence alias, and ABI cases | Five whole-mesh rejection/ledger checks pass |

The independent artifact audit reconstructs all 20 element volumes and the
oriented boundary, verifies all 1,313 CSV rows against the 101 OBJ states,
and checks density-derived nodal mass. Replay and intermediate-step validity
come from the native execution and persistent device flags, not the exported
geometry alone.

The momentum check includes gravity and the accumulated external plane impulse.
The removed-normal-kinetic-energy ledger is `0.27941 J`; it is not total
contact-work closure, because projection also changes elastic and gravitational
potential energy. Sampled mechanical energy never exceeds its initial value.
That observation does not establish a complete energy-balance certificate.

```sh
cmake --build build --target numi-solver-deformable-mesh
./build/numi-solver-deformable-mesh --trajectory build/deformable-mesh-drop
ctest --test-dir build -R '^deformable_mesh.drop$' --output-on-failure
python3 tools/audit_deformable_mesh_trajectory.py build/deformable-mesh-drop
python3 tools/render_deformable_mesh.py build/deformable-mesh-drop build/deformable-mesh-renders
```

Rendering requires Pillow. These are three resolutions of one authored elastic
body on a flat plane. Mesh-resolution convergence, frictional surface contact,
finite-bench edges, measured fruit properties, and two-way contact with the
woven bag remain required before replacing the rigid fruit in the full scene.
The separate full bag timestep-refinement [knot failure](FRUIT_FALL.md#full-scene-half-timestep-qualification-remains-failed)
also remains open.
