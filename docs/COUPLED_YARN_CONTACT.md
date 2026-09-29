# Fruit sharing a yarn contact

The completed finer finite-bench pickup from `6f1e450` fails at frame 163
(1.35833 s). Fruit 3 overlaps yarn 983–984 by **56.7 micrometres**, above
the unchanged **2-micrometre** limit. Both full replays match, and fruits
4, 9, 10 and 11 finish on the lower floor. The 49 regular exported states
pass contact checks, but both peak snapshots fail. The full scene is **FAIL**.
[Completed failure and source fingerprints](assets/finite-bench-load-pickup-96-failure.json)
and [independent peak audit](assets/finite-bench-load-pickup-96-peak-audit.json)
retain the actual result.

A projection trace of the exact saved peak identifies a shared contact:
fruit 3 and bench-supported fruit 10 squeeze the same much lighter yarn.
The separate normal projections alternately resolve one overlap and recreate
the other. Another 160 certificate passes from the exported peak still leave
4.57 micrometres of overlap in that diagnostic. Raising the cap does not
address this mass coupling.

## Simultaneous contact response

The `bbf111a` CPU source solves the small normal contact system for fruits
sharing an airborne yarn. Its contact matrix includes both yarn endpoint
masses, closest-point weights, fruit masses and supported fruit responses.
A bounded active set removes attractive impulses and checks the normal
complementarity residual. The shared yarn moves once with the combined
correction. Exact static arrival limits and blocked fruit support reactions
remain part of the response. A singular or cycling block applies no block
correction and records a fallback to the scalar path.

Yarns already supported on static geometry retain the existing scalar active
set. This change does not approximate their combined static constraints.
The 128-pass certificate cap and existing contact, strain and speed limits
remain unchanged. The certificate and final acceptance now additionally
measure **fruit-pair overlap after yarn corrections**, using the same
1-micrometre solve target and 2-micrometre acceptance limit.

| Exact exported peak diagnostic | Before | After one certificate pass |
| --- | ---: | ---: |
| Fruit/yarn overlap | 56.6926 µm | 0.8510 µm |
| Fruit-pair overlap | 0.0373 µm | 0.0629 µm |
| Local-node overlap | 0.0010 µm | 0.0481 µm |
| Nonlocal yarn, strain and static overlap | Within existing limits | Zero to diagnostic precision |

The independent exported-state audit also passes after projection. Six
mechanics cases cover free and bench-supported fruit with yarn node masses
from 0.00005 to 0.5 kg. They check simultaneous separation, mass-weighted
position balance with the blocked support reaction, zero impulse on an
inactive nearby fruit, and rejection of a singular system without changing
its input. Seven malformed snapshot controls reject.

**These are contact projection checks. They do not advance the saved state
through the full dynamics.** A frozen four-second, 96-substep, two-replay
pickup from this source now completes **PASS**, with five fruit settling on the
lower floor and worst accepted fruit/yarn overlap 0.080 µm. Both complete
481-frame fruit sequences match, and all 51 exported regular/peak states pass.
[Full result and newest video](assets/coupled-bench-pickup-96-evidence.json).
[Source, binary, checks and launch record](assets/coupled-yarn-evidence.json).

```sh
cmake --build build --target numi-solver-cloth-bag
./build/numi-solver-cloth-bag --coupled-yarn-probe
./build/numi-solver-cloth-bag \
  --coupled-yarn-snapshot-probe docs/assets/finite-bench-load-pickup-96-failed-peak.obj \
  --dump-obj build/coupled-yarn-projected.obj
python3 tools/audit_cloth_snapshot.py build/coupled-yarn-projected.obj \
  --require-finite-bench --include-local-node-contacts --include-yarn-static-contacts
```

## Native finite-bench geometry

A separate production geometry helper now implements the authored finite
box and the room floor at **−0.75 m** in FP32 shared with Metal. Native kernels
execute **1,714 cases** twice on Apple M4: 1,707 valid starts cover six faces,
12 rounded edges, eight rounded corners, initial support, high-speed sweeps,
misses and the lower floor; seven invalid inputs reject. Both full native
output arrays match bit for bit. The independent production FP64 geometry,
also used to select random valid starts, finds no hit/feature mismatch.
Maximum impact-position error is **0.394 micrometres** and maximum contact-gap
error is **0.156 micrometres**.

```sh
cmake --build build --target numi-solver-static-geometry numi-solver-static-geometry-metal
ctest --test-dir build -R '^static_contact\.(host|metal)$' --output-on-failure -j1
```

[Actual native geometry output](assets/static-geometry-native.log).
This qualifies geometric casts only. The full native cloth/bench response,
friction, support reactions and spill outcome remain open, as do timestep
convergence, calibrated materials, woven-cloth/elastic-volume coupling and
whole-scene work/reaction closure. The larger native **81,920-tetrahedron**
volume study is independently running under those volume qualification gates.

The native response port in `ed7fc1d` now passes 103 production contact cases
twice and preserves both serialized fruit sequences. Its full four-second
finite-bench candidate stops at frame 4; whole-scene qualification remains open.
[Native response checks, source binding and short-smoke limits](NATIVE_FINITE_BENCH.md).

The `a427ef0` ghost-support repair adds an airborne-friction regression case,
bringing the focused native suite to 104 cases. A new full native replay is
running after both short replays pass the former frame-4 failure.
