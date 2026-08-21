# Recorded six-degree-of-freedom seam grip

The CPU and Metal cloth paths can replay a timestamped hand pose through the
same finite-compliance ten-knot patch on the two-row top cuff. Translation
moves the virtual handle; quaternion rotation rotates each knot's local handle
offset. No cloth particle is pinned or kinematically posed.

This is a deterministic recorded-input interface, not a live mouse or
controller UI. A real experiment still needs synchronized hand tracking,
six-axis seam load, specimen geometry, and held-out outcome measurements.

## File contract

`numi.grip.trajectory.v1` is a relative-pose CSV:

```text
schema=numi.grip.trajectory.v1
time_s,translation_x_m,translation_y_m,translation_z_m,quaternion_x,quaternion_y,quaternion_z,quaternion_w,active
0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,1
```

- Times are finite, nonnegative, and strictly increasing.
- Translation is in solver-world meters relative to the authored seam-handle
  origin.
- Orientation is a right-handed relative unit quaternion in `xyzw` order.
- The first pose must be at `t=0` with zero translation, identity orientation,
  and an active grip. This prevents an unmeasured attachment impulse.
- Version 1 may release once by changing `active` from `1` to `0`; it rejects
  later reactivation because recapturing local offsets from a moving loose cuff
  is not implemented yet.
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
  --grip-trajectory calibration/trajectories/synthetic-rotation.csv \
  --steps 1 --substeps 24 --iterations 32 --replays 2
```

The Metal harness takes the same file. One Metal frame contains 48 persistent
device substeps:

```sh
./build/numi-solver-cloth-metal \
  --grip-trajectory calibration/trajectories/synthetic-rotation.csv \
  --recorded-steps 1 --recorded-dump-every 1 \
  --replays 2 --iterations 32 --strain-sweeps 3
```

Add `--recorded-prefix build/recorded-grab` to export the first replay's OBJ
states. Both executables print the schema, content fingerprint, pose count,
duration, and maximum authored rotation.

## Executable gates

Synthetic coverage requires:

- exact CPU replay for translation-only and translating-plus-rotating inputs;
- different CPU physical-state hashes when only seam orientation changes;
- strict rejection of release followed by unsupported reactivation;
- an ABI-11 Metal grip-rotation probe that matches the independent FP64
  equation, produces nonzero seam displacement, and replays exactly; and
- two exact full-topology Metal replays of the rotating recorded trajectory,
  with zero numerical escape and bounded strain, ground, and self-contact
  residuals.

These gates prove that a recorded six-degree-of-freedom pose reaches the live
cloth equations. The committed trajectories are synthetic and do not prove
that the pose or material matches a physical produce bag.
