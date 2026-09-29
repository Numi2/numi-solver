# Deformable examples: research frontier and measured gaps

The target is a coupled woven bag carrying elastic fruit through gripping,
turning, spill, bench-edge impact and floor settling on Apple silicon. The
September 29 examples do not establish state-of-the-art performance or accuracy.
A passing authored scene is one step toward that target.

## Primary research checked on September 29, 2026

[C-IPC (SIGGRAPH 2021)](https://ipc-sim.github.io/C-IPC/) supplies a relevant
mixed-dimensional benchmark: thick rods, shells and elastic volumes with
coupled contact, friction and strain limiting. Its tablecloth/dinnerware example
is a closer comparison to this bag than an isolated soft-body drop.

[Libuipc / StiffGIPC (2025, project updated August 2026)](https://spirimirror.github.io/libuipc-web/)
is a current GPU implementation reference for coupled rigid bodies, cloth,
threads and soft bodies. Its CUDA/Windows/Linux support does not establish an
Apple-native implementation or performance result for Numi.

[FLASH (April 2026 preprint)](https://arxiv.org/abs/2604.17513) describes GPU-native
nonlinear complementarity contact and deformation constraints. It shows that a
barrier formulation is not the only research direction to compare. Its authors'
RTX 5090 performance and robot results require separate reproduction; they are
not measurements of this code.

[No Free Slide](https://arxiv.org/abs/2308.01696)
and [Geometric Contact Potential](https://arxiv.org/abs/2402.00719) motivate an
additional audit: contact should not produce unrequested tangential resistance
or depend incorrectly on surface tessellation. These papers are comparison
references, not implementation claims for Numi.

## Matched benchmark gates

| Capability | Current evidence | Required next evidence |
| --- | --- | --- |
| Loaded woven bag and finite tabletop | New source `bbf111a`: full CPU96 pickup passes, five released fruit settle on the lower floor, both complete fruit traces match, all 51 saved states pass contacts | Same-source timestep comparison, full native spill, whole-yarn/static contacts throughout the trajectory |
| Native finite contact | Source `a427ef0`: 104 focused response cases pass; a ghost-impulse fix passes the former frame-4 failure and a new full candidate is live | Complete both 480-frame trajectories and audit all retained evidence; qualify continuous whole-yarn collision separately |
| Elastic fruit | Isolated 10,240-tet native body passes authored energy/timestep checks; 81,920-tet study is live. CPU five-owner yarn/elastic-boundary normal contact passes geometry, momentum, angular momentum and represented energy checks | Moving segment/triangle CCD, a common transactional FEM/yarn step, finite-bench contact and complete force/work receipts |
| Mesh and time convergence | Volume spatial comparison fails the 1 mm target. Complete repaired six-second CPU48 passes; CPU96 fails its triangle-area gate despite passing saved contacts. Release sets differ; corresponding fruit positions differ by1.6745m and saved cloth nodes by315.9mm | Repair near-collinear cloth geometry, then match the same physical body, material, load, trajectory and captured times across refinements; report shape, force and work errors |
| Contact/friction consistency | CPU shared-yarn contact block and passive velocity helper pass tested mechanics; two retained energy-injection regressions now dissipate energy | Joint contact, static support, strain and friction residuals; friction-free slide, oblique impact, sliding-to-rolling and separation tests |
| Throughput and closure | Individual native GPU timings and a volume energy ledger exist | Same-workload wall time and simulated time, CPU/GPU ownership, high-percentile frame cost, full grip/contact work and impulse history |

The existing 2 µm authored contact limit, 30 m/s speed limit, 1% volume energy
budget and 1 mm spatial target remain unchanged. Those are local benchmark
budgets, not thresholds borrowed from the cited papers. Cross-engine comparisons
must bind geometry, mass, constitutive law, thickness, friction, timestep,
controller, tolerance, capture cadence and hardware. Aggregate concurrency,
more tetrahedra or smoother rendering cannot substitute for those comparisons.

Implementation proceeds from the actual failing native case, then continuous
whole-yarn bench support and shared contact complementarity, then elastic-fruit
coupling. The accepted CPU scene and unsmoothed native volume stay available
while each larger coupled example earns its own complete evidence.

[Transactional subdivision](TRANSACTIONAL_STATIC_CONTACT.md) now checks whole
interval rollback, exact committed duration and controller/warm-state retention
on CPU. The native adaptive candidate remains unexecuted. The
[reciprocal normal-contact module](ELASTIC_YARN_CONTACT.md) uses both actual yarn
endpoint masses and three owning elastic node masses, including a captured FEM
boundary fixture. It does not advance stresses, detect moving contact or close
thickness-offset frictional torque. Those are the next coupled simulation gates.

[Whole-yarn CPU geometry and velocity evidence](WHOLE_YARN_STATIC_MATH.md)
records all 1,299 seeded sweeps resolved, four extreme-length unresolved
controls and separate normal/friction energy regressions. Those helper checks do not establish
native trajectory collision safety or simultaneous contact closure.
