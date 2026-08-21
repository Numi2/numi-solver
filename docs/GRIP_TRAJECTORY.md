# Recorded six-degree-of-freedom seam grip

The CPU and Metal cloth paths can replay a timestamped hand pose through a
finite-compliance ten-knot patch on the two-row top cuff. Translation
moves the virtual handle; quaternion rotation rotates each knot's local handle
offset. No cloth particle is pinned or kinematically posed.

This is a deterministic recorded-input interface, not a live mouse or
controller UI. A real experiment still needs synchronized hand tracking,
six-axis seam load, specimen geometry, and held-out outcome measurements.

## File contract

`numi.grip.trajectory.v3` is a relative-pose CSV:

```text
schema=numi.grip.trajectory.v3
time_s,translation_x_m,translation_y_m,translation_z_m,quaternion_x,quaternion_y,quaternion_z,quaternion_w,active
0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,1
```

- Times are finite, nonnegative, and strictly increasing.
- Translation is in solver-world meters relative to the authored seam-handle
  origin.
- Orientation is a right-handed relative unit quaternion in `xyzw` order.
- The first pose must be at `t=0` with zero translation, identity orientation,
  and an active grip. This prevents an unmeasured attachment impulse.
- Version 3 may change `active` between attached and released states multiple
  times. Every `0` to `1` transition increments an attachment generation. At
  that transition, the runtime finds the nearest of the 48 live knots in the
  upper cuff row. A strict less-than comparison gives a deterministic
  lowest-ring tie break. The selected center plus two adjacent rings on either
  side defines five knots in each of the two cuff rows, including wraparound,
  for ten unique attachment nodes. The runtime samples their live positions
  before gravity integration, transforms them into the new handle frame,
  resets the grip multipliers, and then resumes the finite-compliance solve.
  This makes the new constraints satisfied up to arithmetic roundoff at
  capture instead of snapping the seam back to its original offsets.
- Re-grab is rejected if any selected cuff knot is more than `0.12 m` from the
  handle. This authored interaction radius prevents a distant handle from
  pulling the bag through space; it is not a measured hand or specimen
  parameter.
- Versions 1 and 2 remain readable for reproducibility. Version 1 rejects
  reactivation after release; version 2 supports reactivation but preserves
  the fixed authored ten-knot patch.
- The pose stream must cover the requested simulation duration. The runtime
  does not extrapolate beyond measured data.

Translation uses piecewise-linear interpolation. Orientation uses normalized
shortest-arc spherical interpolation, with normalized linear interpolation for
nearly coincident samples. Quaternion signs are made continuous on load.
`content_fingerprint` is an FNV-1a content identity for replay matching, not a
cryptographic authenticity claim.

## Replay

The CPU reference uses the `recorded` scenario:

```sh
./build/numi-solver-cloth-bag \
  --scenario recorded \
  --grip-trajectory calibration/trajectories/synthetic-patch-regrab.csv \
  --steps 1 --substeps 24 --iterations 32 --replays 2
```

The Metal harness takes the same file. One Metal frame contains 48 persistent
device substeps:

```sh
./build/numi-solver-cloth-metal \
  --grip-trajectory calibration/trajectories/synthetic-patch-regrab.csv \
  --recorded-steps 1 --recorded-dump-every 1 \
  --replays 2 --iterations 32 --strain-sweeps 3
```

Add `--recorded-prefix build/recorded-grab` to export the first replay's OBJ
states. Both executables print the schema, content fingerprint, pose count,
duration, maximum authored rotation, attachment-generation count, inactive
substeps, capture distance, capture reconstruction error, selection count, and
selected center ring. Metal additionally certifies that every grip carries the
same center marker and that the ten particle indices exactly match the
contiguous two-row patch.

## Executable gates

Synthetic coverage requires:

- exact CPU replay for translation-only and translating-plus-rotating inputs;
- different CPU physical-state hashes when only seam orientation changes;
- backward-compatible v1 rejection of reactivation and v2 rejection of a
  distant re-grab;
- two v2 release/re-grab cycles with continuous local-offset recapture, a
  nonzero inactive-substep count, bounded capture distance, zero or
  roundoff-scale reconstruction error, and exact CPU replay;
- two v3 spatial re-grabs that deterministically move the center from ring 0
  to ring 12 and then ring 24, select ten unique contiguous cuff nodes, reject
  an out-of-radius spatial grab, and replay exactly in CPU FP64;
- an ABI-13 Metal grip-rotation probe that matches the independent FP64
  equation, produces nonzero seam displacement, and replays exactly; and
- two exact full-topology Metal replays of the spatial release/re-grab
  trajectory, with every grip reaching the expected attachment generation and
  center marker, exact selected topology, a GPU-side distant re-grab rejection,
  zero numerical escape, and bounded strain, ground, self-contact, and capture
  residuals.

These gates prove that a recorded six-degree-of-freedom pose reaches the live
cloth equations, including a discontinuous attachment state and spatial patch
change without a positional teleport. The committed trajectories are
synthetic and do not prove that the pose, capture radius, fixed patch width,
finger contact, or material matches a physical produce bag.
