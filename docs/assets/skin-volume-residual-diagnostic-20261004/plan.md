# Synthetic skin mixed-volume residual diagnosis

Date: 2026-10-04. Host: macmini, Apple M4 Pro. This is an engineering diagnostic, not a treatment study or a material-calibration claim.

## Question and current evidence

Does the current mixed-FEM Newton iteration cap cause the v5 synthetic wound-lip traction failure at the seventh 1 ms step, or does the same volume-residual failure persist with more nonlinear iterations? The bound v5 case uses the unchanged 1e-4 volume-residual threshold, default 7 Newton iterations, 10 FGMRES iterations, the 24x8x2 coupon, 10 mN per-bite force, and the full 60-step half-cosine schedule. Its loaded run completed six output rows, then rejected the next step with status 10 at 43 completed microsteps and diagnostics approximately (3e-6, 0, 1e-4, 3e-6). The trial is retained as inconclusive and will not be rewritten.

## Paired diagnostic

Use the exact same source, material, metallib, initial state, mesh, 10 mN dose, timestep, force schedule, contact settings, and existing gates. Run only the first seven outer steps while evaluating the ramp against the original 60-step horizon. Control: 7 Newton iterations. Candidate: 10 Newton iterations. FGMRES remains at its default 10; all residual tolerances remain unchanged. Run sequentially, control first, and stop if it does not reproduce the original failure boundary.

Primary observable: step-7 completion and its mixed volume residual. Prediction: if the outer Newton budget is limiting, 10 iterations will complete step 7 with volume residual <= 5e-5 and all existing determinant, equilibrium, pressure, contact, and mass checks intact. Refutation: the candidate fails at step 7, fails another existing gate, or volume residual remains > 5e-5. Do not escalate to a larger iteration count in this run; inspect the retained Newton/FGMRES diagnostics and revise the solver hypothesis first.

Record source, executable, library, material, and input hashes; raw CSV/logs; per-step GPU time; certificate components; exit code; and SHA-256 hashes of the retained output artifacts. The coupon is one deterministic unit; these matched runs are solver diagnostics, not independent scientific replicates. No claim about calibrated tissue, actual sutures, robots, or clinical behavior follows.
