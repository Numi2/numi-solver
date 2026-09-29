# Native deformable energy accounting

Source `7b47595` adds accepted-step energy ledgers without changing the elastic force law or integrator. The original native level-2 CSV, topology and all 101 OBJ states remain byte-identical. Whole-mesh rejection preserves every work ledger as well as positions and velocities.

The earlier maximum-energy-increase check passes while substantial numerical loss and contact-position work partly cancel. At 1,280 tetrahedra and 100 us timestep, integration loses 0.145573 J while plane projection adds 0.079790 J of total mechanical energy. A kinetic-only contact ledger misses 0.291667 J of elastic-potential change and 0.001707 J of gravity-potential change.

The device now evaluates the exact unconstrained prediction, then the projected state, and separately records elastic/gravity position changes, removed normal kinetic energy, signed integration change, absolute bounds and elastic force dotted with free displacement. These are measurements of the existing discretization, not extra physical forces.

| Timestep | Steps over 0.5 s | Signed free integration change | Absolute position-projection work | Combined error bound / initial energy |
| --- | ---: | ---: | ---: | ---: |
| 100.000 us | 5,000 | -0.145573 J | 0.293487 J | 90.06% |
| 50.000 us | 10,000 | -0.073419 J | 0.147254 J | 45.25% |
| 25.000 us | 20,000 | -0.036789 J | 0.073786 J | 22.67% |
| 12.500 us | 40,000 | -0.018405 J | 0.036886 J | 11.35% |
| 6.250 us | 80,000 | -0.009203 J | 0.018449 J | 5.71% |
| 3.125 us | 160,000 | -0.004610 J | 0.009225 J | 2.92% |
| 1.5625 us | 320,000 | -0.002304 J | 0.004617 J | 1.66% |

All seven complete level-2 timestep traces pass accounting consistency. Each coarse configuration also passes two native replays, its full half-timestep comparison, the independent projection-work probe and five rejected-candidate rollback checks. The largest independent captured-state accounting residual is below 1.2 microjoules. Four negative controls reject missing frames, altered elastic projection work, omitted kinetic loss and nonfinite ledger values.

**The new 1 percent numerical energy budget remains FAIL.** Its conservative bound adds absolute free-integration error and absolute elastic/gravity projection work across steps; it is not the net energy lost by the body. Reducing timestep reduces these errors, but the original energy-increase bound and exact replay do not establish a small energy error. The full 3.125/1.5625 us pair and 10,240-tetrahedron accounting case have now completed with actual exit zero and a verified source/binary manifest. At 10,240 tetrahedra and 100 us, the error bound grows to 144.95 percent; its 50 us comparison remains 73.69 percent. Increasing spatial resolution alone does not close the energy budget.

[Complete source, binary, traces and audit receipt](assets/deformable-energy-evidence.json).

```sh
./build/numi-solver-deformable-mesh --mesh-refinement 2 --timestep 0.000025 \
  --trajectory build/energy-study
python3 tools/audit_deformable_mesh_energy.py build/energy-study --steps 20000
python3 tools/audit_deformable_mesh_energy.py build/energy-study-half --steps 40000
```

Mesh ABI 2 now carries 80 bytes of status. Candidate buffers and their energies are validated before accepting all state and ledgers together. The one-step probe independently reconstructs projection work in FP64 from the actual native pre-projection positions and velocities. The trajectory auditor reconstructs the authored neo-Hookean, kinetic and gravitational energies from every exported node and owning tetrahedron. Native cumulative ledgers cover intermediate accepted steps; exported audits do not independently replay every device step.

## Support-aware velocity Verlet

An optional `--integrator support-verlet` path now splits each force kick around the position advance. Nodes resting on the plane receive their normal support reaction before either kick. This avoids accelerating a resting node into the plane, projecting it out and counting the removed velocity as impact dissipation. Newly arriving nodes still undergo inelastic normal impact.

The post-impact kick evaluates the projected and pre-impact branches at their own positions. ABI 3 separately records their kinetic-energy difference and the additional force-response change beyond explicit normal kinetic removal. The numerical budget adds the absolute force-response change to the earlier integration and position-work bounds.

The first complete 1,280-tetrahedron native run passes the independent 1 percent authored numerical budget: **0.542 percent at 100 us**, versus **90.057 percent for Euler at the identical timestep**. Its 50 us comparison is 0.242 percent. Both methods use the same material, initial surface and masses. Default Euler still reproduces the earlier owning CSV, topology and all 101 OBJ files byte for byte.

Each completed configuration passes two exact native replays, a full half-timestep comparison, six rejected-state/ledger rollback cases and the independent incoming-contact work probe. The resting-contact probe supplies positive impulse with exactly zero kinetic removal, position work and projected kinetic change. The complete 25/12.5 us pair also passes, with bounds of 0.128/0.093 percent. The complete 10,240-tetrahedron study also passes with actual exit zero, exact native replay and a verified source/binary manifest: 0.846 percent at 100 us and 0.330 percent at 50 us, versus Euler bounds of 144.949 and 73.691 percent. Its 1,280-to-10,240-element spatial comparison still fails at 14.927 mm against 1 mm.

```sh
./build/numi-solver-deformable-mesh --mesh-refinement 2 --integrator support-verlet \
  --timestep 0.0001 --trajectory build/support-verlet
python3 tools/audit_deformable_mesh_energy.py build/support-verlet --steps 5000
python3 tools/audit_deformable_mesh_energy.py build/support-verlet-half --steps 10000
```

[Source, binary, traces and independent audit receipt](assets/deformable-support-evidence.json). Five negative controls reject altered contact kinetic work, altered or unbounded force response, a changing integrator and a nonfinite response bound.

This passes a diagnostic numerical budget for the authored drop. It does not certify physical energy closure, measured material or resolution convergence.

Spatial convergence, measured material, reciprocal woven-cloth/volume contact and whole-scene contact-work/reaction closure remain open.
