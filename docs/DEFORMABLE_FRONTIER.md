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
| Elastic fruit | Standalone 81,920-tet native volume completes exact replay and half-timestep checks with 0.254%/0.122% independent numerical energy bounds. A reviewed two-row, five-owner CPU normal block advances all 10,240 elements for 1 µs and removes the prior endpoint closing mode; exact fixed-plane grazing passes 150 cases | Sustained common FEM/yarn advancement, general rotating/incoming CCD, finite-bench contact, native reciprocal coupling and complete force/work receipts |
| Mesh and time convergence | Matched 25 µs 10,240-to-81,920 volume comparison still fails the 1 mm target at 9.34 mm. Complete repaired six-second CPU48 passes; CPU96 fails its triangle-area gate despite passing saved contacts. Release sets differ; corresponding fruit positions differ by 1.6745 m and saved cloth nodes by 315.9 mm | Complete the running shortest-diagonal native pair and audit its spatial error; repair incident cloth area/work ownership, then rerun same-source bag timestep qualification |
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
boundary fixture. The separate [moving geometry candidate](ELASTIC_YARN_MOVING_CCD.md)
now has reviewed interval clearance and contact brackets. A separate
[exact support certificate](ELASTIC_YARN_EXACT_SUPPORT.md) closes the retained
fixed-plane grazing fixture, with independently checked rational/algebraic
material coordinates. General rotation and incoming normal support remain open.
Stress advancement and
thickness-offset frictional torque remain coupled simulation gates.

The [complete captured FEM body](ELASTIC_YARN_FEM_BODY.md) now supplies all
2,057 physical owners and 10,240 source elements for the common CPU step.
Independent exact-rational checks pass every element and rest matrix; all
retained nodal fields and regenerated authoring buffers match. Historical
uploaded inverse-buffer identity and cumulative native status/work checkpoint
remain unavailable. The [reviewed normal-block step](assets/elastic-yarn-fem-normal-block/README.md)
now advances that full body and two authored yarn endpoints together for a
bounded 1 µs CPU interval, with exact replay, two half steps and a positive
5.315 nJ physical energy-balance residual. It is not native continuation,
frictional full-bag contact or long-time qualification. The
[endpoint-aware multistep study](assets/elastic-yarn-fem-endpoint-multistep/README.md)
advances the same complete represented body and locally authored two-node
yarn through 64 linked 20 µs CPU steps to 1.28 ms with exact replay and a
320 µs matched half-step observation. Its fresh public reproduction matches
13 physical output files byte for byte. The closest-witness admission
replaces the fixed centroid test; the +0.322 µJ combined physical energy
residual, CCD, native coupling and full-scene work closure remain open. The
[area admission study](ELASTIC_YARN_AREA_ADMISSION.md) certifies represented
CPU fixture paths while the actual finer-step bag run still fails area.
The [CPU96 incident-work audit](assets/cpu96-area-incident-work/README.md)
finds 15 faces, 10 yarn distances, 14 bends and 10 knots touching the failed
face's owners. On a declared synthetic chord, a locally passive area response
clears those face paths but raises an incident-distance spring proxy by
74.34 µJ; this is not complete work closure or a repaired drop. A private
substep checkpoint and common event journal must precede any coupled impulse.

[Whole-yarn CPU geometry and velocity evidence](WHOLE_YARN_STATIC_MATH.md)
records all 1,299 seeded sweeps resolved, four extreme-length unresolved
controls and separate normal/friction energy regressions. Those helper checks do not establish
native trajectory collision safety or simultaneous contact closure.
