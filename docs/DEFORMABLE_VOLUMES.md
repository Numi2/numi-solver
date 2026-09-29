# From a deformable bag to coupled deformable volumes

The current cloth scenes carry rigid spherical fruit. The next target is fruit
with evolving nodal shape, volumetric stress, and two-way contact with the woven
bag and support plane. A visual change to the sphere renderer is insufficient:
the material, inertia, geometry, and contact response must share the same state.

## Executable volumetric foundation

`numi-solver-deformable-tet` now executes a nonlinear elastic tetrahedron on
Apple Metal. Its pointer-free ABI supplies four current positions, inverse rest
geometry, reference volume, and Lamé parameters in SI units. The device returns
first Piola stress, elastic energy, volume ratio, and all four nodal forces.

The compressible neo-Hookean energy is

```math
W(F)=\frac{\mu}{2}\left(\operatorname{tr}(F^TF)-3\right)
-\mu\log J+\frac{\lambda}{2}(\log J)^2,\quad J=\det F.
```

The corresponding stress is `P = mu F + (lambda log J - mu) F^-T`.
The nodal force matrix is `H = -V P Dm^-T`; the fourth force closes the sum.
This is the compressible material convention documented by
[FEBio](https://febiosoftware.github.io/febio-docs/features/features/solid_material_neo-hookean/).
The authored coefficients are numerical inputs, not measured fruit properties.

The advance kernel integrates elastic forces and gravity on all four nodal
masses. Positions follow the force-integrated velocities; they are not posed.
An invalid or inverted candidate leaves the accepted positions and velocities
untouched. Material, rest-volume, ABI, and nonfinite checks reject malformed
elements rather than substituting another stiffness or repairing their shape.

```sh
cmake --build build --target numi-solver-deformable-tet
./build/numi-solver-deformable-tet --trajectory build/deformable-tet-native
ctest --test-dir build -R '^deformable_tet.constitutive$' --output-on-failure
python3 tools/audit_deformable_tet_trajectory.py build/deformable-tet-native
```

The optional trajectory writes native nodal positions and velocities to CSV
and 51 OBJ snapshots. The three independent elements fall for 0.5 simulated
seconds; compressed and sheared initial states deform from stored elastic
energy. There is no contact surface in this initial volumetric execution.

## Measured M4 result

[Qualification log](assets/deformable-tet-qualified.log) and
[source, binary and trajectory fingerprints](assets/deformable-tet-evidence.json).
The [captured nodal trajectory](assets/deformable-tet-trajectory.csv) retains
all positions, velocities, and masses. Its independent artifact audit verifies
612 CSV rows against all 51 OBJ states and checks positive signed volumes.

| Check | Measured outcome |
| --- | ---: |
| Constitutive cases / rejected malformed cases | 25 / 6 |
| Maximum force disagreement with an independent FP64 energy gradient | 24.19 µN |
| Maximum energy disagreement | 0.173 µJ |
| Maximum net internal force / torque | 0.805 µN / 0.0411 µNm |
| Native motion / refinement | 5,000 / 10,000 steps over 0.5 s |
| Maximum center-of-mass free-fall position error | 3.49 µm |
| Maximum relative nodal shape change | 31.97 mm |
| Maximum relative nodal difference under timestep refinement | 38.77 µm |
| Maximum mechanical-energy drift, including gravity | 0.891 mJ |
| Minimum sampled volume ratio | 0.88614 |
| Two complete native motion replays | Exact at all 51 captured position/velocity states |
| Rejected mass/inversion update | Accepted positions and velocities unchanged byte for byte |

The gradient check perturbs each of the twelve nodal coordinates in FP64 and
differentiates energy, independently of the Metal force implementation. Other
checks cover rigid transforms, force/energy scaling with geometry and material,
net force, net torque, inversion rejection, and exact replay. The dynamics
check also tests every advanced candidate for valid positive volume; the
minimum listed above is sampled at the capture times.
Failure flags accumulate across every device step, so a rejected candidate
between captured frames cannot disappear from qualification. Both full runs
and the refined run recorded zero candidate failures.

## Next simulation gate

This is a running volumetric element foundation, not a coupled soft-fruit scene.
The next implementation must assemble shared-node tetrahedral fruit meshes,
retain their exact boundary surface, distribute contact through the owning
nodal masses, and exchange equal/opposite response with cloth and the plane.
Its qualification must show actual compression and recovery in contact,
positive element volumes, bounded contact residuals and energy error,
timestep refinement, and complete native replay. Measured fruit constitutive
and impact parameters remain a separate calibration requirement.

The new component builds a separate metallib. It does not replace the cloth
source, binary, or library while the corrected full bag replay is running.
