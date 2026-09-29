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

All six complete level-2 timestep traces pass accounting consistency. Each coarse configuration also passes two native replays, its full half-timestep comparison, the independent projection-work probe and five rejected-candidate rollback checks. The largest independent captured-state accounting residual is below 1 microjoule. Four negative controls reject missing frames, altered elastic projection work, omitted kinetic loss and nonfinite ledger values.

**The new 1 percent numerical energy budget remains FAIL.** Its conservative bound adds absolute free-integration error and absolute elastic/gravity projection work across steps; it is not the net energy lost by the body. Reducing timestep reduces these errors, but the original energy-increase bound and exact replay do not establish a small energy error. The finer pair at 3.125/1.5625 us is running from the same frozen source, followed by the 10,240-tetrahedron energy case. They remain pending until their actual terminal results and independent audits.

[Complete source, binary, traces and audit receipt](assets/deformable-energy-evidence.json).

```sh
./build/numi-solver-deformable-mesh --mesh-refinement 2 --timestep 0.000025 \
  --trajectory build/energy-study
python3 tools/audit_deformable_mesh_energy.py build/energy-study --steps 20000
python3 tools/audit_deformable_mesh_energy.py build/energy-study-half --steps 40000
```

Mesh ABI 2 now carries 80 bytes of status. Candidate buffers and their energies are validated before accepting all state and ledgers together. The one-step probe independently reconstructs projection work in FP64 from the actual native pre-projection positions and velocities. The trajectory auditor reconstructs the authored neo-Hookean, kinetic and gravitational energies from every exported node and owning tetrahedron. Native cumulative ledgers cover intermediate accepted steps; exported audits do not independently replay every device step.

Spatial convergence, measured material, reciprocal woven-cloth/volume contact and whole-scene contact-work/reaction closure remain open.
