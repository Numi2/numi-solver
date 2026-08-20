# Cloth realism audit

This file separates corrected defects from still-open physical limits in the
CPU FP64 produce-bag reference. A passing replay does not close an item whose
owning mechanics or evidence is absent.

## Corrected and executable

- The bag bottom is a closed 13 by 13 mapped square weave with 121 interior
  particles and no long single-center fan. The renderer uses the same bottom
  surface instead of leaving the physical cap visually open.
- Authored cloth mass is `0.07805 kg` against `2.33 kg` of fruit. The former
  model was `6.244 kg` and unrealistically heavier than its contents.
- Sphere/cloth position contact uses a bounded connected-patch effective mass,
  preventing three isolated light vertices from producing an explosive local
  response under the realistic load ratio.
- The moving virtual grip target follows an explicit trajectory. Release masks
  latch across the trajectory instead of inspecting only the final frame.
- Fruit carry solid-sphere inertia, angular velocity, and normalized
  orientation. Maximum-dissipation Coulomb impulses couple translation and
  spin at fruit/cloth, fruit/fruit, and fruit/plane contacts; every resolved
  tangent impulse is checked against its friction-cone limit. Grounded fruit
  also have timestep-integrated rolling resistance.
- `--rolling-probe` independently starts a solid sphere sliding at `1 m/s`.
  It reaches the analytic no-slip state `v = r omega = 5/7 m/s`, dissipates
  exactly `2/7` of the initial energy, stays cone-feasible, and replays
  bit-for-bit.
- Predicted cloth/plane and fruit/plane crossings are stopped at the plane
  before iterative contacts. Actual post-contact penetration and the removed
  below-plane free-flight advance are reported separately.
- Conservative advancement solves first time of impact for a moving fruit
  sphere against each linearly moving cloth triangle. The independent
  `--ccd-probe` crosses an entire triangle in one step with both endpoint
  states separated, yet stops at the sphere-plus-cloth contact height and
  replays exactly.
- Cloth self-contact uses graph-derived two-ring topology exclusions over the
  wall and structured bottom, swept plus endpoint vertex/triangle and
  edge/edge constraints, and a final nonlocal primitive-overlap certificate
  below `2 um`. `--self-ccd-probe` crosses both a vertex and an edge completely
  through another cloth primitive in one step and requires exact deterministic
  separation at twice the cloth radius.
- Cloth/cloth contact now applies maximum-dissipation Coulomb friction with
  `mu=0.34` to both vertex/triangle and edge/edge folds. Contact impulses are
  applied in sorted primitive-ID order, conserve linear momentum, and remain
  inside the friction cone. `--self-friction-probe` independently certifies
  exact sticking under sufficient load, cone-limited sliding under lower load,
  energy dissipation, momentum conservation, and bit-identical replay.
- A unilateral strain limiter caps warp, weft, and bottom extension at `28.5%`
  and shear at `38%` while leaving compression and ordinary strain unchanged.
  `--strain-probe` projects a predicted `50%` warp extension to exactly `28.5%`
  without shifting the pair center. Contact/strain reconciliation reports and
  gates both final residuals.
- The grip is a five-node rim patch with finite XPBD compliance, not a pinned
  cloth vertex. All cloth particles remain dynamic; peak attachment force and
  impulse are reported and force above `500 N` fails qualification.
- The authored initial state starts outside the ground plane. Woven-bottom
  extension/compression and a one-cloth-radius plane-correction limit are part
  of the acceptance gate.
- Exported fruit quaternions drive body-fixed surface marks in the README
  renderer, making solver-generated rolling and spin visible without moving
  fruit in presentation code.

## Still open

- The connected-patch contact mass is an explicit approximation, not the full
  coupled deformable response operator targeted for the Metal path.
- Mass, compliance, damping, fruit friction, and geometry are authored
  estimates. Calibration needs a physical specimen, measured load-extension,
  bend, friction, drop, pickup, and spill observations, and held-out replay.
- Rendered yarn interpolation is presentation of the control surface. Contact
  still occurs against cloth triangles, not individual yarn cylinders.

Completion requires owning implementations and executable evidence for every
open item above. Visual plausibility, a GIF, or the current passing gates are
not substitutes.
