# Independent exact grazing review

Reviewed final core SHA256 `f065e1ec62eafe1bf847cc0f89273ec2284312ca6f304289667fd756ef867f27`
and qualification source `7e737fe0be84d1d5d54f0f2b339004f66c79c6d6c237744a1768fa70d8811fab`.
Copies and a uniquely compiled CPU bridge live here. Every run completed within
a 10-second timeout. All final receipt bindings and 18 original/frozen file pairs
were verified unchanged. No production source, GPU or shared build was modified.

## Concrete defect found and repaired

The former `563f6438…` core certified mutable Python coordinates directly as
Fractions, while the FP32 bridge narrowed those same coordinates again. Starting
from the retained motion and adding 4096 to every exposed z coordinate after
construction gave an accepted exact graze at t=1/4 and claimed global
nonpenetration. Its exact proof used yarn z `4096.00400000019`; its actual FP32
owning buffer used `4096.00390625`, against triangle z `4096`, yielding
**93.7501899898052 µm of penetration**. The independent actual-exit-1 failure is
preserved under `history/563f6438-source-binding-failure/`.

The final core takes a private deep snapshot before all admission, oracle and
exact arithmetic. It rejects any owner/radius mutation that differs from FP32.
The identical retained counterexample now returns invalid, accepted false, zero
nodes, and no nonpenetration/normal-closing claim (actual exit 0). The agent's
archived-classifier negative also executes the old core and retains exit 1.
Independent mutation controls additionally reject a 1-m translation with
unrepresented coordinates, malformed shapes/nonfinite values and a double-ULP
radius mutation, while valid represented mutations remain usable. Unsupported
or invalid motion no longer carries an unproved zero normal-closing diagnostic.

## Mathematical checks and retained limits

The closed-plane certificate uses an exact represented normal and verifies at
both motion endpoints that all three triangle vertices share one plane and both
yarn endpoints have the same constant offset d. Since all five paths are linear,
these identities hold for all time. The exact squared comparison
`d² >= radius² * ||n||²` proves global nonpenetration, including an in-plane
triangle deformation/common normal translation. Orthogonal projection subtracts
`d*n/||n||²` before dropping one coordinate, so oblique support is also covered.
The independent review reconstructed these identities for **146 accepted cases**.

With projected initial sets disjoint and the triangle preflight nondegenerate,
first intersection occurs at an endpoint/edge or vertex/segment boundary event.
Collinear features use material-domain boundary roots. Rational roots are exact;
irreducible quadratic roots have exact monotone isolation and ordering. Predicates
reduce modulo the root polynomial to affine expressions, and proportional
polynomials with the same branch identify the same irrational root. Independent
Decimal ordering/predicate controls cover **91 checks**. Two extremely close
branches take 96 counted nodes; a closer pair exhausts exactly **128**, retaining
failure rather than guessing. Preflight nodes, feature enumeration, guessed-root
expansions, ordering refinements and temporal refinements share the same cap;
constant-size exact predicate work is not a separate event node.

Independent material verification reconstructs actual five-owner points:
**109 rational contacts** have exactly radius-squared distance and feasible
simplex weights. For **34 algebraic contacts**, the owning-point squared-distance
numerator minus radius-squared denominator reduces exactly to zero modulo the
irreducible root polynomial; positive-denominator and material-domain bounds
are exact. Algebraic coordinates remain descriptors. Their rounded FP32 values
are not claimed to be contact coordinates or an impulse-ready witness.

The final copied qualification runs exit 0 with **150 cases**, **142 analytic
root comparisons**, two replays per case and at most 13 nodes. All **five**
negative/frontier modes retain exit 1. Independent extra checks cover t=0/t=1,
±1 FP32 ULP support heights, incoming/outgoing normal motion, and fixed-plane
radii down to FP32 minimum subnormal. Below-plane heights and normal motion are
unsupported; rotation remains unsupported even when endpoint supports appear
valid. The 2 µm gate bounds bracket width times the outward motion norm; exact
plane comparison never converts represented penetration into support.

No additional unsafe acceptance was found in these bounded controls after the
repair. This is a restricted, source-bound CPU mathematical certificate for
fixed-orientation parallel support with independently linear owners. It is not
general moving CCD, native Metal or full-scene qualification, a coupled FEM step,
a reciprocal normal/friction impulse, or a momentum/work ledger. Arbitrary plane
rotation and incoming support remain open. These checks are not exhaustive
floating-point, exact-integer resource or physical trajectory proof.
