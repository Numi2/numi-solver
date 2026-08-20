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
- The moving kinematic grip publishes its actual velocity. Release masks latch
  across the trajectory instead of inspecting only the final frame.
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
- The authored initial state starts outside the ground plane. Woven-bottom
  extension/compression and a one-cloth-radius plane-correction limit are part
  of the acceptance gate.
- Exported fruit quaternions drive body-fixed surface marks in the README
  renderer, making solver-generated rolling and spin visible without moving
  fruit in presentation code.

## Still open

- Cloth self-contact is discrete vertex separation. It is not continuous
  edge/edge or vertex/triangle collision and does not prevent all tunneling.
- Sphere/triangle contact is discrete at substep endpoints; there is no swept
  time of impact or continuous collision detection.
- The one-node grip has infinite kinematic authority. A physical pinch needs a
  finite-area compliant attachment plus grip force/impulse reporting.
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
