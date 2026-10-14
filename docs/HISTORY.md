# Historical evidence and separate projects

The active bag entry point is [`./bag`](../bag). Its
[guide](CLOTH_BAG.md) describes current commands and the preserved threaded-yarn
technique. Generated evidence from the earlier work is stored on GitHub,
outside the current checkout.

## Bag and solver archive

[Immutable evidence archive](https://github.com/Numi2/numi-solver/tree/archive/physics-evidence-20261014)
points to `b660979251cdadbc7d19b2a847d580dacd55bcc2`. It retains the original
logs, receipts, failed states, CPU and Metal replays, videos, exploratory FEM
work, old README and the full mechanics/qualification history.

- [Best recorded CPU pickup](https://github.com/Numi2/numi-solver/blob/b660979251cdadbc7d19b2a847d580dacd55bcc2/docs/assets/coupled-bench-pickup-96.mp4): four seconds, two matching replays, five released fruits reach the lower floor.
- [Complete failed native finite-bench pickup](https://github.com/Numi2/numi-solver/blob/b660979251cdadbc7d19b2a847d580dacd55bcc2/docs/assets/native-finite-repaired-pickup-audit/README.md): exact replay, but neither released fruit finishes grounded.
- [Original threaded mechanics](https://github.com/Numi2/numi-solver/blob/b660979251cdadbc7d19b2a847d580dacd55bcc2/docs/CLOTH_BAG.md) and [Metal ownership](https://github.com/Numi2/numi-solver/blob/b660979251cdadbc7d19b2a847d580dacd55bcc2/docs/METAL_CLOTH.md).
- [Exact unfinished local source snapshot](https://github.com/Numi2/numi-solver/tree/archive/cloth-bag-local-20261014): `27fe8567d896864c10ae7b1e98ae708f935bfc73`, preserved before consolidation.

[Download the archived Mac mini bag runs](https://github.com/Numi2/numi-solver/releases/tag/bag-history-20261014).
The release additionally preserves five old source/run copies and an embedded
3,824-file hash manifest. The compressed archive SHA-256 is
`1e6517d544296d40eedec2bcef78b3ab90e5496caa19fbcb44934b5ff4aeb314`.
A fresh extraction verified every file before retirement.

Two small archived geometries also serve as active regression-test inputs in
`tests/fixtures/`; they contain no run logs or media. Historical successes are
bound to their recorded source. They do not qualify later source revisions.

## Skin sutures

Skin-suture traction belongs to [Numi Lab](https://github.com/Numi2/numi-lab),
with a separate source checkout and experiment lineage. It is not a bag scene
and is not executed, built or displayed by `./bag`.

[Historical skin-suture evidence](https://github.com/Numi2/numi-solver/tree/b660979251cdadbc7d19b2a847d580dacd55bcc2/docs/assets/matter-skin-suture-20260930)
remains reachable in the frozen archive because earlier publishing placed it
in this repository. The [published full-horizon traction result](https://github.com/Numi2/numi-lab/blob/8cb11934/numi/SYNTHETIC_SKIN_TRACTION_FULL_HORIZON_RESULT_20261004.md)
is inconclusive: the control completed, while the loaded arm failed after
seven accepted samples. None of its results are evidence for bag mechanics.
