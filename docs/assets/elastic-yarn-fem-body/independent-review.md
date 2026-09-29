# Independent full FEM pack review

The full audit completed with actual exit **0** in about 2.2 seconds. The pack
contains **2,057 nodes, 10,240 positive-volume tetrahedra, 1,280 boundary faces,
40,960 incidence records**, at the captured native frame **38**. No prepared or
production file, GPU state or trajectory was changed. All review writes are here.

## Representation, topology and source binding

All raw ABI sizes pass: three float4 node arrays are each **32,912 bytes**;
10,240 96-byte ABI1 elements are **983,040 bytes**; boundary triples are
**15,360 bytes**; offsets are **8,232 bytes**; incidence records are
**655,360 bytes**. Every coordinate, velocity, mass, inverse-rest, material and
control field is finite and matches `body.json` as represented FP32 bits.
Position/rest w fields and reserved element fields are zero; owning masses and
reference volumes are positive. Element and mesh ABIs remain 1 and 3.

Independent significant-decimal rounding-cell checks recover all **28,798**
frame0/frame38 trace fields uniquely as FP32. They agree with every rest/current
position, current velocity and owning mass bit. All element indices, represented
volumes and material values match the retained topology. Incidence lists are
exactly the complete element/corner-owner gather, with ordered offsets and no
missing or repeated corner. Tetrahedra are distinct, all 10,240 connect through
shared faces, and all 2,057 owning nodes connect. Internal faces have exactly two
owners and opposite orientation; boundary faces exactly one with outward
orientation. The boundary has 642 vertices, 1,920 edges and Euler characteristic
2; every edge occurs once in each direction. Its exact signed volume equals the
exact sum of all tetrahedral volumes. The selected face `[5,1814,1347]` belongs
to element **8768**, with the retained oriented ownership and all seven selected
node fields bit-exact to fixture provenance.

All receipt, retained input and prepared output bindings pass. Each of seven
frozen source/header/kernel/build files matches both the manifest and the actual
Git blob at physics revision `69736313414d3afd81f3b181116870143c13966e`.
An independently compiled, owned CPU binary executes the frozen authoring code
and recreates all six authoring buffers identically. The retained off/on/fast
contraction builds supply additional bound reconstruction evidence.

## Independent geometry and physical values

The audit uses exact binary-rational determinants of represented owner positions
and inverse rows; a separate pivoted FP64 LU evaluation checks det(F). Every
frame38 element passes the unchanged native **J > 1e-6** and scalar reference
consistency **|6V det(invRest)-1| <= 2e-4** limits.

| Quantity | Independent result |
| --- | ---: |
| Minimum det(F) | 0.424337034029949 |
| Maximum det(F) | 1.7175678723338395 |
| Minimum witness | element7296, nodes[7,1249,1251,1687] |
| Maximum FP64 LU versus exact det(F) error | 4.440892098500626e-16 |
| Maximum actual rest-matrix inverse residual | 2.5629844202512686e-7 |
| Maximum exact scalar 6V det(invRest)-1 residual | 2.41048424571176e-7 |
| Total owning mass | 0.3170188445583335 kg |
| Sum of represented reference volumes | 0.00031701884416079906 m³ |
| Exact geometric rest-volume sum, rounded FP64 | 0.0003170188422861788 m³ |
| Mass minus density times represented volume | 3.9753444980306085e-10 kg |

The matrix residual measures `Dm * invRest - I` component by component; it is a
different check from the scalar determinant condition. The owner's FP64 scalar
maximum **2.4104842444305774e-7** agrees with the exact result to arithmetic
rounding. Exact volume summation differs from the owner's ordinary FP64 summation
by about 2.1e-17 m³; this is a rounding difference, not changed physical geometry.
The independent serialized-coordinate volume-ratio minimum agrees with the
retained frame38 geometric audit. The terminal minimum 0.21475289762 covers all
5,000 accepted steps and is not this frame's minimum.

Owning mass is independently reconstructed using the actual FP32
`1000*representedVolume/4` contribution, accumulated in FP64 and cast to FP32;
every nodal mass bit matches. The source's authored **mu3000 Pa, lambda6000 Pa,
density1000 kg/m³** is a consistent compressible Neo-Hookean model with positive
shear/bulk moduli. Its small-strain Young modulus is **8000 Pa**, Poisson ratio
**1/3**, and bulk modulus **8000 Pa**. These are mathematical consequences of the
authored values, not measured fruit/material calibration.

Three independent negative controls retain actual exit **1**: a one-ULP nodal
mass alteration, a truncated element buffer, and a determinant-preserving shear
of inverse-rest rows. The last would pass a determinant-only rest check, but its
matrix inverse residual is **0.0100000064**, so independent geometry rejects it.
The unmodified pack passes both checks; no actual pack defect was found.

## Historical capture limits

Frozen `simulate()` uploads host element records with `newBufferWithBytes`; after
completed command buffers it copies positions, velocities/masses and status to
host `Frame`. The precision12 exporter retains nodal state and topology values,
but omits the inverse-rest rows. The regenerated inverse rows are deterministic
**frozen-source reconstruction** consistent with all captured authoring bits.
There is no independently retained historical uploaded element-buffer dump/hash,
so neither historical inverse-row nor complete uploaded-buffer bit identity is
proved by this audit or the three compiler contraction modes.

Frame38 has **1,900 accepted / 0 rejected steps**, support-Verlet. Its nominal
recorded time is **0.19 s**; the original represented100 µs step product is
**0.18999999520019628 s**. Native frame38 per-element det(F), stress, force and
energy outputs were not exported. The full96-byte native status was not exported:
current/cumulative failure bits and support-impulse/minJ/net-force entries are
missing from the retained CSV. Previous/free/candidate outputs and cumulative
status/work state therefore do not form a bit-exact native continuation
checkpoint. This pack supplies complete physical owning state/material/topology
for a newly implemented CPU step; it does not advance FEM, perform reciprocal
contact, close a work ledger or qualify a coupled elastic-fruit/cloth scene.
