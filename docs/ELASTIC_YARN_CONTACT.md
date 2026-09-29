# Elastic boundary / thick yarn normal contact

The [CPU mechanics module](../include/numi/elastic_yarn_contact.h) resolves reciprocal normal contact between **five independent owning nodes**: two thick-yarn endpoints and three elastic boundary vertices. It uses their supplied positive masses, positions and velocities; it contains no rigid fruit aggregate or fixture mass constants. This qualifies fixed-position, frictionless velocity response on CPU. Moving collision detection, FEM advancement, Metal execution and a complete elastic-fruit/bag scene remain open.

[Source/binary evidence](assets/elastic-yarn-contact-evidence.json) binds the executed header, scalar dependency, probe, unique binary, compiler and actual exits. The default [random mechanics log](assets/elastic-yarn-contact-host-random.log) passes 12,008 geometry queries and 4,001 impulse rows; the [captured fixture log](assets/elastic-yarn-contact-host-fem.log) passes 12,009 and 4,003 respectively. The fixture run includes 2,001 rows where all five owners move.

| Maximum error or change | Measured | Required limit |
|---|---:|---:|
| Distance against independent FP64 geometry | 0.1733 µm | 2 µm |
| Velocity against FP64 row algebra | 1.895 µm/s | 10 µm/s |
| Impulse against FP64 row algebra | 48.63 nNs | 100 nNs |
| Actual-mass linear momentum residual | 52.50 nNs | 100 nNs |
| Actual-mass angular momentum residual | 1.095 nNms | 10 nNms |
| Remaining closing speed | 0.820 µm/s | 10 µm/s |
| Accepted represented kinetic gain | 0 J | 0 J |

## Query and response contract

`numiElasticYarnClosestSegmentTriangle` considers endpoint/triangle features, all three segment/edge witnesses and interior axis/triangle crossing. It handles parallel and zero-length segments. Nonfinite input, a nonpositive radius or a degenerate triangle returns an invalid witness. A crossing/coincident axis is detected but has no unique unsigned-distance normal: the velocity solve returns `NumiElasticYarnAmbiguousNormal` with `valid=false`. The 1,160 ambiguous geometric cases in the suite are detected cases, not accepted impacts. Qualification covers the tested metre-scale geometries and the captured centimetre-scale fixture; it does not establish unbounded scale/translation accuracy.

For row weights `c=[1-u,u,-b0,-b1,-b2]` and normal `n`, set `g_i=c_i*n`. The module evaluates `k=Σ dot(g_i,g_i)/m_i`, `vrel=Σ dot(g_i,v_i)`, `J=max(0,-vrel/k)` and `v_i'=v_i+J*g_i/m_i`. The normal follows the reconstructed axis-to-triangle witness vector, so the reciprocal impulses are central. Geometry is checked independently by an FP64 constrained quadratic program with KKT active-set enumeration. The FP64 velocity oracle separately uses the represented FP32 witness; long-double energy and independent actual-mass momentum sums inspect the represented response.

`numiElasticYarnWitnessAtCallerValidatedCoordinates` and `numiElasticYarnResolveCallerValidatedNormal` support explicit material-point rows on a flat parallel closest set. **The caller must establish physical manifold membership.** These functions check coordinate-domain and represented geometry validity; they do not certify that arbitrary supplied coordinates identify a closest or physically valid manifold row. One row does not resolve an entire parallel interval or adjacent-face manifold.

`numiElasticYarnCanPublish(result)` requires `valid=true` and a solved, inactive or separated status. Every failed or energy-rejected response returns all original velocities and zero accepted impulse; candidate velocities remain local until validation succeeds. `energyUpper` is an attempted-candidate diagnostic when status is `NumiElasticYarnEnergyRejected`, and must not be booked as accepted contact work. Input owning buffers are never mutated.

The outward-rounded kinetic-change bound prevents positive or nonfinite represented candidate changes from being accepted. Four negative controls show that endpoint-only queries miss interior crossing, omitted triangle inverse masses add 202.625 J, frozen triangle response leaves a 1 Ns momentum residual, and FP32 rounding adds 2,606.669 J in a deliberately extreme common-velocity control. That last candidate is rejected, and its returned state is unchanged. Ten invalid-input and three explicit status controls pass. The [initial isolated failure log](assets/elastic-yarn-contact-isolated-initial-failed.log) retains 438 barycentric failures: replacing `1-v-w` with `1-(v+w)` fixes negative rounded weights without lowering targets. The original isolated source and evidence are preserved.

## Captured owning-node fixture

The [tiny five-node fixture](assets/elastic-yarn-fem-frame38-five-nodes.txt) and [provenance receipt](assets/elastic-yarn-fem-frame38-provenance.json) retain a real elastic boundary face from physics source `69736313414d3afd81f3b181116870143c13966e`: 2,057 nodes / 10,240 tets, frame 38 at 0.19 s. Face `[5,1814,1347]` belongs to exactly one source tet. Its recorded positions, velocities and masses—38.70, 77.40 and 61.92 mg—are unchanged. A local 4 mm-radius yarn with two authored 50 mg endpoint masses is constructed above that face; it is not captured full-bag yarn or calibrated material evidence.

The explicit interior row is within 1.40 nm of the independent global distance minimum. Its [recorded response](assets/elastic-yarn-fem-frame38-contact-result.json) moves all five owners, applies 62.1069 µNs and removes 31.0534 µJ of actual kinetic energy. It does not repair the authored 1 µm overlap or advance the FEM stress.

## Run the CPU probe

The CMake target and both host tests are also available:

```sh
cmake --build build --target numi-solver-elastic-yarn
ctest --test-dir build -R '^elastic_yarn\.(normal_host|fem_boundary_host)$' --output-on-failure
```

The standalone [probe](../tools/elastic_yarn_contact_probe.cpp) needs no fixture for its random mechanics suite. Build a separate binary:

```sh
set -e
numi_elastic_probe_dir=$(mktemp -d)
xcrun clang++ -std=c++20 -O3 -fno-fast-math -ffp-contract=off -Wall -Wextra \
  -Iinclude tools/elastic_yarn_contact_probe.cpp \
  -o "$numi_elastic_probe_dir/numi-solver-elastic-yarn-contact"
"$numi_elastic_probe_dir/numi-solver-elastic-yarn-contact"
"$numi_elastic_probe_dir/numi-solver-elastic-yarn-contact" \
  --fixture docs/assets/elastic-yarn-fem-frame38-five-nodes.txt
```

`--fixture-output PATH` optionally writes the detailed fixture response; default execution writes only JSON to stdout. The fixture format is one radius followed by five rows of `x y z vx vy vz mass` in SI units. The optional fixture qualification checks the supplied local parallel/centroid row against the global FP64 minimum; arbitrary five-node geometry need not satisfy that specific fixture contract.

## Native coupling still required

The current volume owner is `tools/deformable_mesh.mm::simulate`, using accepted/free/candidate positions and `velocityAndMass[node].w`, with force gathering, inversion validation and commit in `src/metal/DeformableMesh.metal`. Yarn endpoints belong to `NumiClothBagGPUParticle`; segment IDs resolve through `NumiClothBagGPUDistance.particlesAndColor`. Existing sphere/yarn contact records do not represent three independently moving FEM vertices.

A native coupled step needs a boundary-face index buffer, records mapping both yarn owners and all three elastic owners, unclipped free predictions for both bodies, conservative moving segment/triangle CCD, explicit unresolved/degenerate query status and deterministic shared-node manifold scheduling. It must reevaluate candidate FEM stress and second-kick forces, then validate and publish both buffer sets and all ledgers together. The ledger must retain five-owner impulses, linear/angular residuals, kinetic change, elastic/gravity position work, force-response work and signed plus absolute/positive bounds. This CPU velocity bound does not certify that full step.

Thickness-offset friction generally requires yarn rotational degrees of freedom or an explicit torque/couple to close angular momentum. Friction, moving CCD, positional work, full FEM advancement, native ABI/Metal execution, spatial convergence and full-scene qualification are not provided by this module.
