# Six-second loaded-bag drop

This synthetic stress input increases the scene beyond fruit falling from a
held bag. The loaded deformable bag is carried past the finite tabletop edge,
the compliant seam grip turns 180 degrees, and the grip releases at 3.8 s.
Gravity then advances the freely moving cloth and twelve rigid fruit toward
the lower floor, with 2.2 seconds remaining for impact and settling. The grip
turn is not a prescribed rotation of the whole bag. No cloth or fruit path is
posed, no fruit is given a release impulse, and no re-grab is requested.

The authored scene retains 1,465 dynamic cloth nodes, 2,904 yarn segments,
5,754 local node contacts, twelve sphere fruits, 0.07805 kg cloth and 2.33 kg
fruit. The finite slab is 1.5 by 1.0 m, 80 mm thick, with top at z=0 and the
room floor at z=-0.75 m. Material, gravity, friction, rolling resistance,
contact thickness, compliance and acceptance gates are unchanged.

## Authored seam motion

The [121-pose CSV](../calibration/trajectories/finite-bench-loaded-drop.csv)
uses the version 3 relative-pose contract. A cubic smoothstep is sampled at
50 ms intervals; the runtime interpolates translation and unit quaternions.
All translations are relative to the authored cuff handle, in metres.

| Time | Input |
| --- | --- |
| 0 s | Attached, zero translation, identity rotation |
| 0–1.2 s | Lift the seam by 1.2 m |
| 1.2–1.6 s | Hold |
| 1.6–3.0 s | Carry 1.1 m in +x, beyond the tabletop edge |
| 1.6–3.2 s | Turn the grip 180 degrees about the world y axis |
| 3.2–3.8 s | Hold the final pose |
| 3.8–6.0 s | Grip inactive; cloth and fruit continue dynamically |

```sh
./build/numi-solver-cloth-bag --scenario recorded --finite-bench \
  --grip-trajectory calibration/trajectories/finite-bench-loaded-drop.csv \
  --steps 720 --substeps 48 --iterations 32 --replays 2 \
  --dump-frames build/loaded-drop --dump-every 10 \
  --dump-obj build/loaded-drop-final.obj \
  --dump-knot-peak build/loaded-drop-peak.obj \
  --knot-trace build/loaded-drop-knot.csv \
  --fruit-trace build/loaded-drop-fruits.csv
python3 tools/audit_cloth_snapshot.py build/loaded-drop-*.obj \
  --include-local-node-contacts --require-finite-bench
```

## Evidence status and required outcomes

The trajectory parses through the production loader and a two-frame startup
passes two exact replays, including its three accepted frame-state hashes.
That is only input/startup validation. A complete six-second CPU48 job is
running from frozen source `4186cfa`, binary and trajectory; its actual terminal
result is pending. It must finish both 720-frame replays. The complete input
fingerprint is SHA-256
`1313ab661da38fe49c764209391e489050aef5e35399230561d3be7a8c62afd1`.

Whole-scene physical gates, the ordered hash of every accepted frame, all
73 exported contact audits and the final inactive grip must pass. The loaded
drop outcome additionally needs measured cloth COM descent after release and
actual lower-floor cloth support. A solver PASS without that outcome is not a
qualified loaded-bag drop. Saved cloth states and all 721 per-fruit frame
states in both replays supply the witnesses. Report collisions, folding,
re-entry and any failed gate from the actual run; do not replace them with
posed deformation or a shortened execution.

This remains an authored woven-cloth/sphere model. The native elastic volume
benchmark is separate, and its mesh-resolution convergence is still open.
Replacing sphere fruit with deformable volumes requires reciprocal surface
contact and cloth/volume coupling. Native finite-bench execution, calibrated
specimens and full contact-work/energy closure remain open.
