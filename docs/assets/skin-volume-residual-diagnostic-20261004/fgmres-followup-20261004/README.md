# Skin-traction follow-up: stricter KKT solve

**Outcome: inconclusive; the volume-residual gate still rejects step 6.** This
single solver-policy candidate kept the synthetic coupon and all physical
settings and final acceptance limits fixed. It is not a treatment study,
material-calibration result, or clinical example.

## Candidate and result

The prior matched 7-versus-10 Newton diagnostic ruled out the Newton cap as
the sole cause. This follow-up tested a tighter inexact-KKT solve: Newton
remained at 7, the FGMRES budget rose from 10 to 20, the scheduled-forcing
amplitude changed from 0.25 to 0.0625, and the forcing floor changed from the
square root of the equilibrium tolerance to the unchanged tolerance itself.
All material, mesh, initial state, load schedule, timestep, residual gates,
contact rules, and other physics settings stayed fixed. The registered
prediction was step-7 completion with the volume residual at or below `5e-5`.

| Run | Last accepted mean gap | Volume residual at last sample | Terminal status | FGMRES work | Wall time |
|---|---:|---:|---|---:|---:|
| Prior 7-Newton control | 0.594221 mm | 8.3524e-5 | Step 6, code 10, exit 1; 43 completed microsteps | 8 | 27.44 s |
| Stricter KKT candidate | 0.586943 mm | 9.3535e-5 | Step 6, code 10, exit 1; 41 completed microsteps | 9 of 20 allowed | 37.93 s |

Both runs accepted six samples and rejected the same next state with reported
diagnostics approximately `(3e-6 equilibrium, 0 correction, 1e-4 volume,
3e-6 pressure)`. At the last accepted frame the bite force was 0.176 mN per
site. The candidate's gap changed by 13.06 µm from the 0.600000 mm starting
gap, versus 5.78 µm for the matched control. This 2.26× response difference is
a solver-sensitivity result on one synthetic coupon; it does not establish
which trajectory is more physically accurate. The stricter solve did not
clear the unchanged gate, and FGMRES stopped after 9 iterations rather than
exhausting the new budget. Raising the iteration budget alone is therefore
not supported as the next change.

## Candidate clip

[![Six accepted frames of the stricter KKT candidate, marked inconclusive and
showing the unchanged volume gate](stricter-kkt-accepted-prefix-inconclusive-poster.png)](stricter-kkt-accepted-prefix-inconclusive.mp4)

The clip renders the candidate's six accepted 1 ms states at 1 fps. The wound
detail uses actual geometry at a stated ±0.8 mm vertical range; the gap chart
uses a fixed 0.55–0.61 mm range and places samples against the original 60 ms
horizon. There is no frame interpolation or deformation exaggeration. The
footer marks the step-6 rejection and states that no wound closure is shown.

## Reproduction record

- Preregistered question, prediction, and stop rule: [`plan.md`](plan.md).
- Exact input hashes: [`input-SHA256SUMS`](input-SHA256SUMS).
- Candidate source snapshots: [probe](source/apps/synthetic_skin_wound_traction_probe.mm)
  and [runtime](source/matter/src/runtime.mm).
- CPU-only layout/preflight output: [`metadata/preflight.log`](metadata/preflight.log).
- Raw command, process exit, runtime log, accepted samples, and lip geometry:
  [`output/`](output/).
- Rendering source: [`output/render_candidate.py`](output/render_candidate.py).
- [`SHA256SUMS`](SHA256SUMS) covers this follow-up folder; the parent report
  retains the matched Newton control and earlier run.
- Host: Apple M4 Pro Mac mini. Source base commit:
  `eed22b54e93db18ff8d2bf36b5c454a52efdab79`. Diagnostic probe SHA-256:
  `ec0a2b6fa97444fb190ac558115fe69afd28afc62413b285175648d61d7d36c3`;
  runtime SHA-256:
  `a3a5e51b70bdb362f2797c8b7ff29925b416d498e7acca145f03b30c155d907f`.

Next work should capture per-Newton and per-Krylov residual history at the
rejected microstep, then inspect mixed pressure/volume coupling and
globalization under the same gates. Do not increase the FGMRES budget again
without evidence that its current cap is binding. No needle, thread, robot,
measured tissue, healing, clinical qualification, or 100× performance claim is
present.
