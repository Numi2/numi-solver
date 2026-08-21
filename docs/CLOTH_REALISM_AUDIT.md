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
- Air loads replace the former global linear and angular velocity dampers.
  Yarn drag resolves cylinder crossflow and axial skin friction from segment
  length and orientation; fruit drag resolves projected area and rotational
  surface drag. A dissipative time-integrated quadratic update replaces the
  timestep-dependent force cutoff. The probe matches analytic force and
  torque, verifies kinetic-energy removal, spatial subdivision invariance,
  exact isolated coarse-versus-refined velocity, a zero-force co-moving case,
  and exact replay.
- Cloth/plane friction replaces the former fixed horizontal decay with a
  maximum-dissipation Coulomb impulse capped by the reconstructed normal
  reaction. Fruit rolling resistance replaces load-independent exponential
  angular decay with a normal-load-dependent torque cap that preserves
  vertical spin. Independent probes cover loaded, sticking, sliding, and
  zero-load cases.
- A unilateral limiter caps axial extension at `28.5%` while leaving
  compression and ordinary compliant strain unchanged. Contact/strain
  reconciliation certifies the final residual.
- The grab is a finite-compliance ten-knot patch spanning five neighboring
  knots on each of the two folded top-cuff rows, not a pinned point or a bottom
  handle. All 1,465 knots and twelve fruits stay dynamic; attachment force and
  impulse are measured and gated. The two cuff rows have authored local bend
  reinforcement while the body yarn remains soft.
- Release masks latch across the trajectory and are distinct from numerical
  escape. Classification uses all 48 top-rim knots, an outward-oriented mouth
  frame, the projected opening polygon, and full-sphere clearance through the
  cap or around every 3D rim segment after a mouth-crossing candidate. It no
  longer relies on distance to one bottom particle. A `25 mm` clearance
  hysteresis rejects grazing/re-entry chatter. The focused probe certifies
  contained, grazing, through-cap, edge-exit, far-outside, and rotated-mouth
  cases.
- The qualified two-second pickup replay lifts the seam patch, snaps it
  downward, and records four robust sphere exits from the 48-knot mouth; all
  four remain spilled on the plane. It has
  `escaped_mask=0`, sub-micrometer published contact/strain residuals, and a
  bit-identical second replay; no fruit path is prescribed. The same motion
  retains plural release and all physical gates at 96 substeps. The old
  24-substep pickup is explicitly rejected for contact/strain residual and
  single-release outcome, so it is not visual evidence.
- Exported fruit quaternions drive the visible body marks. The rasterizer draws
  only the exported axial yarn graph and grip connectors; it does not pose the
  bag or fruit.

## Still open: specimen fidelity

- Mass distribution, axial and knot compliance, bend compliance,
  aerodynamic coefficients, contact friction, rolling resistance, yarn
  diameter, bag geometry, and grip motion are authored estimates. They are not
  measurements of the pictured bag.
- Each simulated yarn is one control-scale capsule between lumped knot masses.
  The model does not resolve multi-filament twist, fiber migration, knot
  tightening or sliding, plastic set, creep, damage, moisture, or rate- and
  load-dependent textile behavior.
- Calibration requires a physical specimen and recorded load-extension, bend,
  friction, drop, seam-pickup, swing, and spill trials. Parameters must be fit
  on calibration trials and then judged on held-out trajectories using measured
  geometry, force, timing, and release outcomes.
- The complete trajectory evidence remains an FP64 CPU reference. The initial
  Metal cloth ABI now owns full-topology gravity integration, graph-colored
  axial XPBD, crossing-angle knots, ground-aware yarn bending, the unilateral
  strain ceiling, ten-knot seam grip, twelve-fruit translation, all 66
  fruit-pair normal contacts, cloth/fruit ground projection, all 34,848
  present/swept fruit-yarn candidate geometries, conservative-advancement CCD,
  deterministic ground-aware sphere/yarn normal response, accumulated normal
  impulse publication, and velocity publication. That subset passes
  independent FP64 equation comparison and exact replay; see
  [METAL_CLOTH.md](METAL_CLOTH.md). Yarn/yarn contact, sparse contact
  compaction, all friction transactions, aerodynamics, fruit rotation, release
  classification, complete-trajectory parity, profiler evidence, and matched
  physical outcomes remain open.

Completion requires the specimen measurements and held-out replay above.
Visual plausibility and the current executable certificates are necessary but
not substitutes.
