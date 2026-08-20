# Cloth realism audit

This file separates executable mechanics from the physical evidence that is
still missing. A passing replay or convincing GIF does not calibrate the model
to a real produce bag.

## Corrected and executable

- The bottom is a closed 13 by 13 square weave mapped one-to-one onto the
  48-node wall boundary. Only the folded top mouth is open.
- The live mechanics use 2,904 axial yarn capsules, 1,369 compliant warp/weft
  crossing-angle knots, and 2,834 three-knot bend chords. Diagonal render
  triangles are absent from stretch, bend, fruit contact, and self-contact.
- Authored cloth mass is `0.07805 kg` against `2.33 kg` of fruit. Every
  capsule contact uses the true inverse masses and closest-point weights of its
  two endpoint knots.
- Fruit contact is sphere versus moving yarn capsule. Conservative advancement
  catches a complete one-step crossing, while the direct response probe
  certifies free center-of-mass conservation, supported transfer, first-plane
  activation, exact separation, zero ground violation, and replay.
- Yarn self-contact is continuous capsule/capsule contact with graph-derived
  two-ring exclusions, swept first impact, endpoint resolution, local knot
  particle closure, and a completed-frame overlap certificate below `2 um`.
- Maximum-dissipation Coulomb impulses couple translation and rotation for
  fruit/yarn, fruit/fruit, fruit/plane, and yarn/yarn contact. Focused probes
  certify analytic slide-to-roll, sticking, cone-limited sliding, momentum
  conservation, dissipation, cone feasibility, and exact replay.
- A unilateral limiter caps axial extension at `28.5%` while leaving
  compression and ordinary compliant strain unchanged. Contact/strain
  reconciliation certifies the final residual.
- The grab is a finite-compliance five-node attachment to the top opening seam,
  not a pinned point or a bottom handle. All 1,465 knots and twelve fruits stay
  dynamic; attachment force and impulse are measured and gated.
- Release masks latch across the trajectory and are distinct from numerical
  escape. The qualified pickup releases three fruits through the open mouth
  with `escaped_mask=0` and a bit-identical second replay.
- Exported fruit quaternions drive the visible body marks. The rasterizer draws
  only the exported axial yarn graph and grip connectors; it does not pose the
  bag or fruit.

## Still open: specimen fidelity

- Mass distribution, axial and knot compliance, bend compliance, damping,
  friction, rolling resistance, yarn diameter, bag geometry, and grip motion
  are authored estimates. They are not measurements of the pictured bag.
- Each simulated yarn is one control-scale capsule between lumped knot masses.
  The model does not resolve multi-filament twist, fiber migration, knot
  tightening or sliding, plastic set, creep, damage, moisture, or rate- and
  load-dependent textile behavior.
- Calibration requires a physical specimen and recorded load-extension, bend,
  friction, drop, seam-pickup, swing, and spill trials. Parameters must be fit
  on calibration trials and then judged on held-out trajectories using measured
  geometry, force, timing, and release outcomes.
- The current evidence is an FP64 CPU reference. A production Metal cloth path
  still needs owning kernels, parity against these transactions, deterministic
  replay, profiler evidence, and matched physical outcomes.

Completion requires the specimen measurements and held-out replay above.
Visual plausibility and the current executable certificates are necessary but
not substitutes.
