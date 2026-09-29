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
iterations and two full replays from a frozen build. Its actual exit is **1**.
The maximum frame-end dynamic speed is **27.1111 m/s**, below the unchanged
30 m/s gate. It still fails: the reported minimum frame-end render-triangle
area is approximately **6 × 10⁻⁹ m²**, below the unchanged **10⁻⁸ m²** gate.
The log rounds area to nine decimal places. Independent 60-digit Decimal
evaluation of all 218,880 triangles in the 76 retained states finds a robust
witness at **frame 620, triangle 2813, nodes 1433/1444/1445**: serialized area
**5.828993541 × 10⁻⁹ m²**. Including coordinate serialization uncertainty,
its possible original area is **[5.1862, 6.4718] × 10⁻⁹ m²**, entirely below
the gate. The three bottom-grid owners are nearly collinear against the floor.

[Independent area audit](assets/loaded-drop-corrected-96-area-audit.json) ·
[Exact retained witness](assets/loaded-drop-corrected-96-area-witness.obj).
This is the sole definite numeric violation visible in the rounded summary.
Unexported final velocities and exact quaternion norms do not allow a claim
that every other runtime predicate independently passed. The solver updates
area at frame ends; retained captures do not bound intervening substep area.
The retained prior 35.93 m/s cloth-speed failure remains a failure
against the unchanged 30 m/s target; focused controls do not establish that
this defect caused that full-scene peak. Peak speed frame, body and a complete
cloth velocity snapshot are now retained whenever that unchanged gate fails.

The repaired 48- and 96-substep runs have now each completed both full
six-second fruit replays. Identical frozen source, binary, controller and gates
give **different release outcomes**: no fruit is released at 48 substeps;
fruit 10 is released at 96, with a saved floor contact at 4.8 seconds. Their
maximum corresponding fruit position difference is **1.6745 m**. Saved cloth
nodes differ by **315.895 mm** at frame 660, node 1018; the largest mass-weighted
RMS difference is **215.843 mm**, also at frame 660. Release bits first differ
at frame 537. This is a **timestep-convergence failure**, even though both
complete fruit replay pairs match exactly and their static geometry passes.

[Exact first-replay comparison](assets/loaded-drop-corrected-first-replay-comparison.json) ·
[48-substep fruit trace](assets/loaded-drop-corrected-48-r1-fruits.csv) ·
[96-substep fruit trace](assets/loaded-drop-corrected-96-r1-fruits.csv).
The 48-substep invocation now has actual exit 0, two complete replays and
passing audits of all 73 regular snapshots, both replay peaks and the final
copy. Cloth center descends 1.7041 m; 33 nodes finish at the lower floor. It
releases no fruit. The [complete 48-substep evidence](assets/loaded-drop-corrected-48-terminal-evidence.json)
binds that authored result. The 96-substep run completes with matching ordered
721-frame hashes and exact fruit replay, but retains its actual failure. All
73 regular states, both contact peaks and the final copy pass independent
contact audits. Cloth descends 1.7538 m; 31 nodes finish at the lower floor,
along with fruit 10.

[Actual CPU96 terminal evidence](assets/loaded-drop-corrected-96-terminal-evidence.json) ·
[Complete matched comparison](assets/loaded-drop-corrected-complete-comparison.json) ·
[Complete CPU48 fruit replays, gzip](assets/loaded-drop-corrected-48-complete-fruits.csv.gz) ·
[Complete CPU96 fruit replays, gzip](assets/loaded-drop-corrected-96-complete-fruits.csv.gz).

The floor observation is a saved
radius and vertical-velocity match; it does not qualify reaction or energy
closure. The failed finer run does not qualify a replacement full-scene video.

This is CPU FP64 contact bookkeeping. Full native finite-bench spill,
continuous whole-yarn contact through every accepted substep, reciprocal
elastic-fruit/bag coupling, material calibration, temporal/spatial convergence
and complete contact-work/reaction histories remain open.
