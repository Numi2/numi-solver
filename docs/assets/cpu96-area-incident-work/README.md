# CPU96 area incident-work audit

Run the CPU-only reproduction from any checkout on a machine with `clang++`
and Python 3.11 or newer:

```sh
python3 docs/assets/cpu96-area-incident-work/reproduce.py
```

The runner checks the exact source/input hashes, copies this package to a
fresh temporary directory, compiles `topology.cpp` against the included frozen
cloth source, and runs the 15-face synthetic-chord audit. It retains all
logs, actual exits, JSON outputs, and a `reproduction.json` in the printed
directory. The compiled harness is generated only in that private directory;
no compiled binary is published. The frozen cloth source matches SHA-256
`d575b0154af35b2a110b21fc091c9913955b0ee62637ef67ba3c09cb0b623f9b`.
The historical complete CPU96 binary is identified by SHA-256
`1c6d90eb01d8ad49394c426433b2b19e7ade2dc5395fc4ab675336cdd3c19458`
in its retained manifest, but that binary is not needed or included here.

The real `makeCloth` topology has 1,465 nodes and 2,880 render faces. Failed
face 2813 has owners [1433,1444,1445], each 5e-5 kg. Those owners touch
15 faces, 10 distance/yarn constraints, 14 bends, 10 knots, and 23 local-node
contact candidate pairs. Their yarn-distance graph reaches every cloth node.
The 10 segments have up to 120 fruit-segment pairs across 12 fruit, but active
contacts require actual dynamic history and are not inferred from topology.

The frame-0-to-frame-620 interpolation is explicitly a **synthetic one-second
chord**. The retained OBJ files give positions only; their intermediate owner
history and velocities are unavailable. Exact-rational casting finds face
2813 as the only threshold event among the 15 incident faces on this chord.
The isolated, previously reviewed area response clears the 15 represented
post-response affine paths and loses `1.43475546513e-10 J` of local kinetic
energy. A diagnostic spring proxy from the 10 authored incident distance
rest lengths and compliances rises `7.43385526859e-5 J`. A separate
Decimal80 calculation agrees with the FP64 proxy's sign and within `1e-10 J`.
The bend proxy rises `1.15903829886e-10 J`. These are **partial proxies**,
not the source's complete XPBD or contact work: knot, grip, fruit, self,
ground, friction, gravity and aerodynamic terms are absent. Therefore local
kinetic passivity cannot certify whole-cloth physical closure. The explicit
`--claim-repaired-drop` control exits 1.

The source mutates cloth/fruit positions, XPBD multipliers, contact ledgers,
metrics, grip state and release state within a substep; area is currently
measured only after the frame's substeps. The smallest defensible next source
change is a global private substep trial with exact rollback of every owned
state and deferred output. It must journal every represented owner path after
each mutating solver operation and find the common earliest event over an
authored material-face set. Only after that transaction works can a coupled
area response and full K/material/contact/external-work ledger be tested.
The corrected 720-frame CPU96 drop still fails the strict area gate; this
package does not repair it or establish a new full drop, spatial convergence,
material calibration, or native GPU qualification.

`source-manifest.json` binds the runnable inputs. `published-evidence.json`
binds the preserved outputs and their actual exits. The original ignored
local audit and its exact source/fixture/CPU96 failure bindings are archived
in `historical-evidence.json`; historical paths there are provenance, not
required by the portable runner.
