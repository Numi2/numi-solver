# Complete FEM body for coupled yarn experiments

The next yarn/volume experiment uses the complete captured **2,057-node,
10,240-tet** native body at frame 38, recorded time **0.19 s**, from physics
revision `6973631`. Its accepted positions, velocities, positive lumped masses,
source rest positions, material, reconstructed inverse-rest rows, connectivity,
**1,280 boundary faces** and **40,960 incidence entries** are retained together.
This is a prepared input body; no new time advance is claimed by this pack.

[Complete payload manifest](assets/elastic-yarn-fem-body/payload-manifest.json) ·
[Preparation evidence](assets/elastic-yarn-fem-body/owner-evidence.json) ·
[Independent audit](assets/elastic-yarn-fem-body/independent-review.md).

## Source and owning state

Seven frozen owning source files match revision `6973631` and the earlier
terminal manifest. The exact extracted host authoring function reconstructs
all six authoring buffers. Contraction off/on/fast builds produce identical
buffers; every rest position, mass, connectivity, represented volume and
material entry agrees with the retained trace/topology bits. Independent decimal
rounding-cell checks uniquely recover **28,798** selected FP32 trace fields.

The selected contact face **[5,1814,1347]** belongs solely to tet **8768**.
Its three positions, velocities and masses agree bit for bit with the
[published five-owner fixture](ELASTIC_YARN_CONTACT.md). The body uses the
source's authored compressible Neo-Hookean material: μ = **3,000 Pa**,
λ = **6,000 Pa**, density = **1,000 kg/m³**. Material calibration is still open.

| Quantity | Independent result |
| --- | ---: |
| Total owning mass | 0.3170188445583335 kg |
| Sum of represented rest volumes | 0.00031701884416079906 m³ |
| Minimum reconstructed det(F) at frame 38 | 0.424337034029949 |
| Maximum reconstructed det(F) at frame 38 | 1.7175678723338395 |
| Maximum rest-matrix inverse residual | 2.563 × 10⁻⁷ |
| Maximum scalar 6V det(inverseRest) consistency error | 2.411 × 10⁻⁷ |

Exact-rational determinants and independent pivoted FP64 evaluation agree
within 4.441 × 10⁻¹⁶. All elements pass unchanged native limits
**det(F) > 10⁻⁶** and scalar rest consistency **≤ 2 × 10⁻⁴**. The oriented
boundary and all tetrahedra have exactly equal geometric rest volume; the
connected surface has Euler characteristic 2. Independent negative controls
retain exit 1 for an altered mass, truncated ABI and determinant-preserving
inverse shear. A scalar determinant check alone would miss the last defect.

## Complete represented payload

The nine gzip payloads decompress to the reviewed bytes, including a complete
indexed `body.json` and raw little-endian ABI arrays. Node state/rest records
are float4; elements are 96-byte ABI-1 records; incidence entries are uint4.
No body owner or tetrahedron is replaced by a simplified fixture.

```sh
python3 docs/assets/elastic-yarn-fem-body/unpack.py
```

The unpacker verifies compressed and original hashes and extracts into a fresh
directory under `build`. Preparation and audit sources/receipts remain available
with their original path bindings. Unpacking checks payload identity, rather
than rerunning the historical native simulation.

Inverse-rest rows are a deterministic frozen-source reconstruction. The
historical uploaded inverse buffer was not separately exported. Frame-38 native
per-element stress/J outputs and the cumulative status/work checkpoint are also
absent. These inputs support a newly implemented common CPU step; they do not
establish a bit-exact native continuation.

## Common step required next

A coupled step must evaluate the complete continuum, gather its forces, apply
reciprocal yarn contact to actual owning states, advance both bodies, re-evaluate
stress/contact and validate the complete candidate before committing it. Gravity,
support reactions, projection work, contact dissipation and numerical residual
must remain separately observable. Continuous collision coverage, friction,
material calibration, convergence and complete native bag/volume execution
remain open.
