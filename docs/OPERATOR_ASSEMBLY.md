# GPU response-column operator assembly

## Contact response equation

For contact `i`, owner `o`, and its local generalized coordinates, the input
contains

```math
J_{i,o}\in\mathbb{R}^{3\times d_o},
\qquad
V_{i,o}=M_o^{-1}J_{i,o}^T\in\mathbb{R}^{d_o\times3}.
```

The assembly kernel constructs every present 3x3 contact block as

```math
W_{ij}=R_i\,\delta_{ij}+
\sum_{o\in\mathcal O_i\cap\mathcal O_j}J_{i,o}V_{j,o}.
```

This is the complete sparse `J M^-1 J^T + R` action. Contacts sharing a rigid
body or articulation owner therefore receive off-diagonal coupling; they are
not solved as independent local pairs.

## Deterministic packed contract

Each contact owns a strictly owner-sorted term span. A term carries a stable
owner index, DOF count, packed `3 x DOF` Jacobian range, and packed `DOF x 3`
response range. The maximum is 32 owner terms per contact and 32 DOFs per
term. The authored block-CSR topology remains strictly source-sorted.

One SIMD32 group owns one island and one lane owns one target contact. The
lane advances monotonically through its CSR row and merge-joins target/source
owner spans. Each shared owner is visited once. For each DOF, one Jacobian
value updates all three source axes with ordered FP32 fused multiply-adds.
There are no floating-point atomics, append races, lane-0 matrix assembly, or
unordered reductions.

The kernel rejects:

- invalid capacities, ranges, term ordering, owner IDs, or DOF counts;
- missing diagonal blocks;
- any missing or extraneous shared-owner CSR coupling;
- nonfinite inputs or assembled coefficients;
- authored regularization that is asymmetric or not positive-semidefinite;
- assembled `W_ij`/`W_ji^T` disagreement above the FP32 symmetry tolerance.

## Transaction and command-buffer ownership

The assembly kernel writes candidate coefficients but begins with an invalid
output solver header. Only a completely valid, symmetric operator commits a
versioned `NumiTemporalConeStreamHeader`. The streamed solver is encoded in a
second compute encoder on the same caller-owned command buffer and consumes
that header directly. A rejected assembly therefore makes the chained solver
publish typed failure and zero impulse output without host intervention.

There is no CPU readback, second command queue, internal command-buffer commit,
or wait between response-column assembly and contact solve.

## Positive-semidefinite authority

Symmetry and the PSD principal-minor certificate for every authored 3x3
regularization block are checked on GPU. Full-operator positive
semidefiniteness then follows from the provider contract `V=M^-1J^T` with
positive-definite mass response. Arbitrary symmetric response columns are not
silently declared physical. The qualification harness independently
reconstructs every dense operator and applies FP64 Cholesky before accepting
the generated batch.

The generic assembly ABI still consumes authored topology, Jacobians, and
response columns. The rigid response kernel generates rigid 6-DOF terms from
body/contact/material data. The articulated response adapter consumes the
canonical articulated operator's analytic world-point Jacobian and checked
lower Cholesky factor, rotates the Jacobian into the contact frame, and solves
three triangular systems per contact to obtain `M^-1J^T`. Both paths generate
positive diagonal implicit contact-law regularization on GPU. Collision
detection and topology creation remain upstream responsibilities.
