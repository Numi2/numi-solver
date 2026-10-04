# Synthetic skin volume residual: tighter KKT solve diagnostic

Date: 2026-10-04. Host: macmini, Apple M4 Pro. This is a bounded engineering diagnostic on one synthetic coupon, not a treatment study or material-calibration claim.

## Question and prior result

The prior matched diagnostic kept all gates fixed and compared seven versus ten Newton iterations. Both stopped at outer step 6 with mixed-FEM nonlinear-solver status 10, 43 completed microsteps, 8 FGMRES iterations, and residual diagnostics approximately (3e-6 equilibrium, 0 correction, 1e-4 volume, 3e-6 pressure). Accepted sample columns and lip geometry were identical. This refuted the Newton cap as the cause.

The runtime source reveals that its current inexact-Newton forcing floor is sqrt(relative equilibrium tolerance), while its scheduled forcing is 0.25 * 2^-min(Newton index, 4). With the default 1e-4 equilibrium tolerance, the floor is 1e-2. The revised hypothesis is that the KKT solve accepts a correction too early to reduce the mixed volume residual below its separate unchanged 1e-4 acceptance gate.

## Preregistered candidate

Reuse the prior seven-Newton control as the matched reference. In one isolated source worktree, make one stricter linear-solve policy candidate: keep Newton at 7; increase the total FGMRES budget from 10 to 20; change the forcing schedule amplitude from 0.25 to 0.0625; and set the forcing floor to the existing relative equilibrium tolerance instead of its square root. This targets a fourfold tighter forcing schedule without changing any final residual gate, material, load, mesh, initial state, contact settings, or timestep. Run only the first seven outer steps while the load is still evaluated on its original 60-step half-cosine horizon.

Primary prediction: the candidate completes outer step 6, records a mixed volume residual <= 5e-5, and passes every unchanged determinant, equilibrium, pressure, contact, mass, and transport gate. Refutation: it repeats the step-6 rejection, fails another existing gate, or completes with volume residual > 5e-5. Record the actual FGMRES work and wall/GPU time. Run one candidate only; if refuted, inspect its trace before changing another setting.

The failure boundary remains a failure. Do not relax tolerances or promote a partial run as wound closure. Record source, executable, dylib, metallib, material, and plan hashes plus raw logs, samples, geometry, command, exit code, and output hashes. This deterministic paired solver diagnostic is not an independent scientific replicate and supports no calibrated tissue, clinical, robotic, suturing, or 100x-performance claim.
