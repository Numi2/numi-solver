# CPU finite-bench impact and departure repair

A cloth knot or fruit can hit the bench and then depart under constraint motion
within the same substep. The old CPU path reconstructed velocity from the
pre-impact chord and retained an old support normal after departure. Focused
face, edge, corner and lower-floor controls show 4.7 to 5.0 m/s velocity errors
and spurious friction capacity at a final 20 mm clearance.

The repaired path removes incoming velocity at the exact swept impact, retains
that physical velocity, and reconstructs subsequent constraint displacement
from the post-impact prediction. An unsupported final position clears current
support and friction capacity. The repair changes no trajectory, speed cap,
damping or acceptance budget.

All 16 face/edge/corner/floor impact and departure cases pass twice with exact
replay. Maximum analytic velocity error is 1.2644e-13 m/s; departure support
impulse is exactly zero. Disabling only the impact velocity update fails by
10 m/s. Existing finite-bench trajectory and finer-timestep, sustained support,
coupled fruit/yarn, local-node, self-friction and deformable-response checks
pass. [Source and raw qualification](assets/cpu-impact-departure-repair-evidence.json)
and the [frozen source/binary manifest](assets/cpu-impact-departure-launch-manifest.json)
bind the result.

A new complete six-second recorded drop uses 96 substeps, 32 constraint
iterations and two full replays from a frozen build. Its qualification is
pending. The retained prior 35.93 m/s cloth-speed failure remains a failure
against the unchanged 30 m/s target; focused controls do not establish that
this defect caused that full-scene peak. Peak speed frame, body and a complete
cloth velocity snapshot are now retained whenever that unchanged gate fails.

The repaired 48- and 96-substep runs have now each completed their first full
six-second fruit replay. Identical frozen source, binary, controller and gates
give **different release outcomes**: no fruit is released at 48 substeps;
fruit 10 is released at 96, with a saved floor contact at 4.8 seconds. Their
maximum corresponding fruit position difference is **1.6745 m**. Release bits
first differ at frame 537. This is a **timestep-convergence failure**, even
though both saved fruit traces pass independent static geometry checks.

[Exact first-replay comparison](assets/loaded-drop-corrected-first-replay-comparison.json) ·
[48-substep fruit trace](assets/loaded-drop-corrected-48-r1-fruits.csv) ·
[96-substep fruit trace](assets/loaded-drop-corrected-96-r1-fruits.csv).
The 48-substep invocation now has actual exit 0, two complete replays and
passing audits of all 73 regular snapshots, both replay peaks and the final
copy. Cloth center descends 1.7041 m; 33 nodes finish at the lower floor. It
releases no fruit. The [complete 48-substep evidence](assets/loaded-drop-corrected-48-terminal-evidence.json)
binds that authored result. The 96-substep second replay and terminal audits
remain pending. The floor observation is a saved
radius and vertical-velocity match; it does not qualify reaction or energy
closure. No new full-scene video is qualified by these partial results.

This is CPU FP64 contact bookkeeping. Full native finite-bench spill,
continuous whole-yarn contact through every accepted substep, reciprocal
elastic-fruit/bag coupling, material calibration, temporal/spatial convergence
and complete contact-work/reaction histories remain open.
