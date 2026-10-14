# Numi Solver

Apple Metal physics for rigid bodies, articulated systems and deformable yarns.

## Threaded cloth bag

```sh
./bag
```

This builds the Metal runtime and opens the native bag window. The default
swing carries twelve spherical fruits in the original woven-yarn bag. The
window displays computed states as they arrive, then lets you play or scrub
the exact exported replay. Closing the window stops a running worker.
Simulation is computed offline; playback does not imply real-time physics.

[Run the bag, choose a scene, and understand the yarn technique](docs/CLOTH_BAG.md).
[Archived experiments and the separate skin-suture work](docs/HISTORY.md).

The bag uses the existing 1,465 dynamic knots, 2,904 axial yarn capsules,
compliant crossing knots, bending and a ten-knot cuff grip. Render triangles
never replace yarn collision or mechanics. All physics runs on Metal.

Other solver interfaces: [mathematics](docs/MATHEMATICS.md),
[rigid mechanics](docs/RIGID_MECHANICS.md),
[articulated mechanics](docs/ARTICULATED_MECHANICS.md),
[operator assembly](docs/OPERATOR_ASSEMBLY.md).
