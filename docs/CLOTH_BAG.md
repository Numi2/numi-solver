# Threaded cloth bag

From the repository root, run `./bag`. The same entry point builds, runs,
checks and views the bag. Full Xcode with the Metal toolchain, CMake and Python
3 are required on an Apple-silicon Mac.

```sh
./bag                                # complete 0.5-second cuff swing, two replays
./bag run --scene pickup              # four-second lift/snap and settling
./bag run --scene rest                # one-second unsupported rest on the plane
./bag run --scene swing --headless    # compute without opening a window
./bag view                           # reopen the latest computed replay
./bag check                          # focused native mechanics gates
```

The default scene remains the original threaded technique: **1,465 dynamic
knots**, **2,904 axial yarn capsules**, **1,369 crossing-angle constraints**
and **2,834 bend chords**, carrying twelve rigid spheres. A closed woven
bottom joins the side wall; the top cuff is open. The virtual hand grips ten
massive cuff knots through finite-compliance constraints. No cloth or fruit
trajectory is imposed. The authored controller moves only the cuff target.

Metal owns gravity, yarn stretch/bend/knot mechanics, sphere/yarn contact,
yarn self-contact, ground response, friction, air loads and fruit rotation.
The triangles in OBJ files are presentation topology. They are never used as
a substitute for the yarn capsules in mechanics or collision. The AppKit
viewer renders exported knot positions and fruit poses without changing them.

The window shows states while Metal computes. Afterwards, Play/Pause and the
slider operate on the recorded trajectory. This is an offline simulator with
native replay, not a real-time interactive physics claim. Both complete
replays retain ordered state hashes and their final buffers must match exactly.
The original independent oracle/probe harness remains available through
`numi-solver-cloth-metal`; ordinary scene launch does not run that whole suite.

Each run writes to a fresh `build/bag/runs/` directory. `build/bag/latest`
points to the newest run. It contains actual knot/fruit snapshots, fruit
telemetry, the native log, source/binary/input hashes and `bag-scene.json`.
Incomplete or rejected runs keep their accepted states and return failure.
No new run silently overwrites an earlier run.

Scene completion is distinct from spilling. The receipt reports both
`result`/`complete` and `plural_spill_and_landing`. A completed lift that keeps
its fruit is not a demonstrated spill. The historical pickup-spill acceptance
gates remain unchanged in the research harness.

For a custom recorded cuff motion:

```sh
./bag run --scene recorded --grip calibration/trajectories/synthetic-patch-regrab.csv --frames 1
```

[The recorded-pose contract](GRIP_TRAJECTORY.md) supports translation,
quaternion rotation, release and nearest-cuff re-grab. It remains a point-hand
controller, not a simulated finger grasp.

The production scenes use the established support plane. `--finite-bench`
opts into the separate experimental finite tabletop and lower-floor contact
path; the latest full-topology smoke rejects at frame 2 and rolls back.
It is kept out of the default launch. Calibration and broader physics research
remain described in the [immutable historical archive](HISTORY.md).

Source owners: `tools/cloth_bag_metal.mm`, `src/metal/ClothBag.metal`,
`include/numi/cloth_bag_gpu.h`. `tools/cloth_bag.cpp` is the separate CPU FP64
reference. Skin sutures are maintained in Numi Lab, outside this bag scene.

## Verified on 14 October 2026

The full default swing completes **60 frames / 0.5 seconds twice** on Apple
M4 Pro, with matching ordered state hashes, all 61 OBJ states and complete
fruit traces. Both runs have zero native failure flags. All 61 saved states
pass independent 2 µm contact checks; worst sampled overlap is 0.063 µm.
Matching knot-pair distances change by up to 0.272 m, demonstrating deformation.
The final launcher also passes its two-frame M4 replay with the same first
states. Six focused Metal gates, three relocated-fixture tests, native viewer
controls/event loop, atomic export and worker cancellation checks pass.

This default swing does not spill fruit. Its receipt explicitly records that
outcome; it does not turn the older pickup/landing failure into a pass.
[Current replay and source-bound verification](https://github.com/Numi2/numi-solver/releases/tag/threaded-bag-20261014).
