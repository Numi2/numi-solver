# Independent preventive area-admission review

The frozen prototype admits a velocity response on a triangle whose area is
exactly the strict `1e-8 m²` gate. Reproducer: owners
`((0,0,0),(1,0,0),(0,2*G,0))`, `G=1e-8`, all velocities zero, all masses
one. Frozen `passive_velocity` returns `accepted=true`; its `cast` correctly
returns `initial_area_violation`. The separate candidate changes only the two
helper inequalities from `<` to `<=`. The original frozen source and receipt
are unchanged. The direct strict-gate check exits 1 on frozen and 0 on the
candidate. The copied full qualification exits 0 and writes a byte-identical
`result.json` to the frozen qualification. The copied qualifier only changes
its study-root path so it can run from this separate directory.

The independent review exits 0. It checks 48 analytic earliest roots of
planar flips, a terminal root at t=1, a repeated tangent at t=1/2, and four
roots of a quartic trajectory. Exact direct affine-owner cross products agree
at the analytic roots and at outward brackets; the maximum bracket width is
`2^-44`. It checks 64 general 3D clear motions using a separate conservative
lower bound on one cross-product component. For 36 admitted seeded velocity
responses, an independent exact rational calculation confirms nonpositive
actual represented kinetic change and bounded momentum and angular residuals.
The local-rate-only future collapse, moving pin, exhausted proof budget, and
mutable-input controls reject or preserve state as specified.

This supports the corrected candidate as an isolated CPU reference primitive,
not as a drop repair. `preventive` currently reaches the velocity helper only
after its own strict boundary-prefix check, so the frozen defect does not by
itself demonstrate a false `preventive` admission. The primitive still has no
common earliest event across incident faces and constraints, no authority over
the native integrator's represented path and saved velocities, and no ledger
for all affected yarn, bend, contact, and external work. The corrected CPU96
full drop remains FAIL; direct integration into the cloth authority would
therefore be premature. See `receipt.json` for exact source hashes and exits.
