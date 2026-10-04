# Trace-guided tightening of the synthetic wound-lip traction solve

Date: 2026-10-04. Host: Mac mini, Apple M4 Pro. Scope: one bounded, synthetic FEM solver candidate; no tissue calibration, clinical, or physical-robot claim.

The trace-only reproduction repeated the step-6 failure exactly. At the failing control step, all 42 attempted microsteps formed nine FGMRES columns, crossed the current relative forcing threshold of 0.03125, and ended between 0.0249 and 0.03117. The 20-column cap was not binding. The unchanged certificate still reported a volume residual at its 1e-4 rejection boundary.

Test one trace-guided candidate: lower only the scheduled forcing amplitude from 0.0625 to 0.015625. Since the recorded Newton index is 1, this changes the active target from 0.03125 to 0.0078125. Keep the forcing floor, Newton budget (7), FGMRES budget (20), material, mesh, input state, load schedule, timestep, and every acceptance gate fixed. Preserve complete per-column FGMRES trace and exact terminal status floats.

Primary prediction: the first seven control steps complete, step 6's mixed volume residual is <= 5e-5, and every unchanged gate passes. Refutation: the solver exhausts its budget, the same volume gate rejects, or any other existing gate fails. Run this candidate once only. If refuted, inspect its residual trace and exact failure status before proposing another change. The load continues to use its original 60-step half-cosine schedule.

A separate cardiac Metal workload is still running on the shared Mac mini GPU. The user asked to proceed; therefore record the concurrent workload and do not use this run's wall/GPU time as isolated performance evidence. The numerical output and trace remain separate artifacts. The residual history is a least-squares estimate, not an independently recomputed true residual.
