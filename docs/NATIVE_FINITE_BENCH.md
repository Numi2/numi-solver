# Native cloth against a finite bench

Source `ed7fc1d` gives the native cloth kernels the same authored box and
lower floor as the CPU scene: bench bounds **[−.75, −.5, −.08] to [.75, .5, 0] m**,
with the room floor at **−.75 m**. `--finite-bench` selects this experimental
trajectory mode; the existing plane oracle probes retain their plane mode.

The production kernels pass **103 focused cases twice** on Apple M4 with
bit-exact physical states. They cover six faces, 12 rounded edges, eight
rounded corners and the floor. Both 4 mm yarn-radius bodies and 50 mm fruit
bodies receive radial impacts. Separate checks exercise sustained support,
Coulomb sliding, rolling resistance, supported fruit pairs, and a yarn that
lifts a node away from support without retaining bench friction.

| Production response check | Maximum error | Unchanged limit |
| --- | ---: | ---: |
| Position | 0.0746 µm | 2 µm |
| Velocity | 0.0119 mm/s | 1 mm/s |
| Support impulse | 3.825 µNs | 100 µNs |
| Independent static penetration | 0.0143 µm | 2 µm |

The independent FP64 impact reference casts the actual represented FP32 ray.
Rounding a world-coordinate start does not preserve a perfectly radial ray
at a 4 mm edge. Sustained support, friction, rolling and pair checks use
analytic physical expectations. The separate **1,714-case** native geometry
probe still passes twice with no hit/feature mismatch and maximum impact
position error **0.394 µm**.

## Response and replay changes

Static casts now form edge/corner normals in feature-local coordinates.
The native advance retains its position and velocity after static impact;
finalization reconstructs later constraint motion against that prediction.
This avoids feeding the chord across a curved impact into tangential velocity.

Finite support projects fruit-pair, fruit/yarn and cloth friction responses
against their actual surface normal. Blocked positional reactions remain in
kg*m until division by timestep; blocked velocity reactions are impulses in
Ns. Surface friction and rolling resistance follow the local normal. A body
that leaves support cannot retain a bench velocity clamp or friction capacity.
These changes require **ABI 15**, with 96-byte particles and 144-byte fruits.

Each trajectory now keeps **both fruit sequences and both OBJ sequences**.
Replay one keeps the historical OBJ names; replay two uses `-r2-`. The finite
combined fruit CSV includes replay identity, independently reconstructed static
clearance and current support impulse. The auditor accepts its exact 17-column
schema and the existing 16-column CPU schema, and rejects incomplete sequences,
duplicates, reordered columns and invalid impulses.

```sh
cmake --build build --target numi-solver-cloth-metal
./build/numi-solver-cloth-metal --finite-bench-probe
./build/numi-solver-cloth-metal --finite-bench \
  --pickup-prefix build/native-finite-pickup --pickup-steps 480
python3 tools/audit_replayed_finite_fruit_trace.py \
  build/native-finite-pickup-fruits.csv --expected-frames 480 \
  --require-released-landings
```

## Qualification boundary

The frozen **two-frame smoke** lasts only **0.0167 s**. Its two complete short
fruit sequences match, and all six exported states pass independent fruit,
yarn, local-node and whole-yarn/static contact audits. Its actual runner exit
is **1**, with `complete=false` and `result=FAIL`: it does not satisfy the
480-frame spill test and has no released fruit.

The first full `ed7fc1d` candidate stops at **frame 4** in both replays. Its
original report incorrectly labels the requested 480 frames as complete; the
captured CSV contains only frames 0–4. A read-only per-dispatch observer finds
the first failure in fruit/bench friction at substep 7 of that frame and matches
all ten saved states and both fruit sequences exactly against the unobserved run.

Source `a427ef0` fixes the cause: scaling equivalent free/supported responses
before subtracting them creates a **2.386e-10 Ns** ghost support impulse on an
airborne fruit. The blocked response is now subtracted before impulse scaling.
An added airborne-friction case produces **exactly zero** static impulse; all
**104 focused cases** pass. A deliberate one-expression regression control
reproduces the defect and fails that case. Both five-frame diagnostic replays
now pass the former failure point with zero failure flags and exact replay.
Their actual runner exit remains 1 because five frames do not qualify the full
480-frame spill. Completion reporting now includes actual captured frame counts,
failure flags and the failing saved state.

A new frozen **four-second, two-replay, 48-substep** candidate from `a427ef0` is
running. Full native spill outcome and contacts throughout its substeps remain
pending. Focused passing probes and saved-state audits do not establish joint
static/contact/friction closure, temporal convergence, calibrated materials,
elastic-volume/cloth coupling or whole-scene work/reaction closure. The
published normal impulse is current support friction capacity; it is not the
full history of static impulses after a body departs support. The
81,920-tetrahedron volume study remains independent.

[Actual early failure, read-only observer comparison, positive/negative controls,
repair source and new launch](assets/native-finite-ghost-impulse-repair.json).

[Source, frozen payload, actual logs, short-state audits and launch receipt](assets/native-finite-bench-response-evidence.json).
