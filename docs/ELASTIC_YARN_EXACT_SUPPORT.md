# Exact grazing support between five deformable owners

The reviewed CPU certificate resolves the retained later graze at **exact
t = 1/4**. Its yarn parameter is 1/2 and triangle weights are (0,0,1), with
**zero relative normal closing speed**. The broader interval candidate still
returns unresolved just before that event. This separate exact certificate
passes **150 cases**, including **142 analytic grazing roots**, with two exact
replays and at most **13/128 nodes**. The 2 µm temporal/spatial gate is unchanged.

[Reviewed source](assets/elastic-yarn-exact-support/closed_support.py) ·
[Owner receipt](assets/elastic-yarn-exact-support/owner-evidence.json) ·
[Independent review](assets/elastic-yarn-exact-support/independent-review.md) ·
[Independent receipt](assets/elastic-yarn-exact-support/independent-review-evidence.json).

## The admitted geometry

Exact rational arithmetic uses the represented FP32 owners. All three triangle
owners must remain in one plane with a fixed orientation, and both yarn owners
must retain the same strictly positive separation from that plane. These exact
identities at both endpoints hold throughout the independently linear paths.
The squared separation must equal the represented squared yarn radius; larger
separation gives clear, while smaller separation is unsupported even by one ULP.
In-plane triangle deformation and common normal translation are covered.

The certificate projects orthogonally before dropping a coordinate. Nine
projected boundary features produce exact orientation polynomials of degree at
most two. Exact material-domain predicates, root isolation and root ordering
select the first feasible intersection, including collinear entry and isolated
double-root tangency. The plane identity proves global nonpenetration; absence
of earlier projected roots proves the clear prefix.

Rational material coordinates are exact. Irrational roots retain algebraic
coordinate descriptors whose squared distance and simplex bounds are proved
at the root. They are not rounded coordinates ready for an impulse. An oblique
normal (0,3,4) control reaches t = 1277/5120, while another reaches sqrt(1/8).
Outward FP64 time brackets and an outward owner-motion bound provide the spatial
width. The largest measured width is 3.141 × 10⁻¹⁶ m.

## Independent checks and retained failures

The independent reviewer reconstructed **146 exact plane identities**,
**109 rational** and **34 algebraic** owning-point contacts, and **91 root
predicate/order checks**. A close-root control needs 96 nodes; a still closer
pair exhausts exactly 128 and rejects. Node accounting includes face admission,
feature visits, enclosure, ordering and bracket refinements. Arbitrary-precision
scalar work is additional cost; this is not native constant-cost throughput.

Review found a real state-binding defect in the earlier prototype. Mutating
public Python coordinates after construction let its exact proof use different
geometry from the FP32 bridge. Translating the retained fixture by 4096 m hid
**93.750 µm of represented penetration**. The repaired solver validates exact
FP32 representation and takes a private deep snapshot before all proof and
oracle operations. Twelve owner controls and fifteen independent binding/radius
controls pass, including a caller mutation during bridge admission.

All five rejected-classifier/frontier runs retain actual exit 1: a nonfinite
legacy plane, rounded shallow penetration, endpoint-only rotating support,
the unsupported rotating-plane frontier, and the actual archived mutable-owner
classifier. These are retained failures. The immutable owner receipt predates
the completed review; the separate reviewer receipt binds the final core
`f065e1ec62eafe1bf847cc0f89273ec2284312ca6f304289667fd756ef867f27`.

## Reproduce the public pack

On Apple silicon macOS, from the repository root:

```sh
python3 docs/assets/elastic-yarn-exact-support/reproduce.py
```

The runner verifies all published copies, compiles a CPU bridge into a fresh
directory under `build`, runs the 150 cases, five expected exit-1 modes and both
independent controls, then records actual exits and hashes. Historical receipts
and logs remain byte-identical. Public reproduction does not claim to reverify
unavailable historical binary/capture paths; that check is separately retained
in the owner and independent receipts.

## The next coupled simulation gate

Arbitrary plane rotation, incoming/outgoing normal motion and penetrating
crossings remain outside this primitive. General moving CCD, reciprocal
response, friction, FEM stress advancement, work/reaction closure and native
complete-scene execution remain open. The broader
[moving interval candidate](ELASTIC_YARN_MOVING_CCD.md) and
[instantaneous normal-response module](ELASTIC_YARN_CONTACT.md) remain distinct
components. A common owned FEM/yarn step must re-evaluate continuum stress and
contact on its advanced state before publishing both bodies together.
