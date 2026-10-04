# Synthetic skin traction: volume-residual diagnosis

**Outcome: inconclusive skin traction; solver-budget hypothesis refuted.** This
is a deterministic engineering diagnostic on one synthetic coupon, not a
calibrated tissue result, complete suture simulation, or treatment study.

## What was tested

The prior v5 run used one synthetic 24 × 8 × 2 mixed-FEM coupon (2,304
tetrahedra), a 10 mN per-bite-force endpoint, a 60-step half-cosine schedule,
1 ms frames, and the unchanged `1e-4` volume-residual gate. It accepted six
samples, then stopped at outer step 6 with Matter status code 10 after 43
completed microsteps. The final reported residual vector was approximately
`(3e-6, 0, 1e-4, 3e-6)`; the third value is the mixed volume residual.

The preregistered question was whether the seven-iteration Newton cap caused
the rejection. The paired diagnostic ran only the first seven outer steps but
evaluated both force sequences on the original 60-step horizon. It kept the
same material, metallib, mesh, initial state, timestep, load schedule, FGMRES
budget, and acceptance thresholds. The only changed solver setting was the
Newton iteration budget.

| Newton budget | Accepted samples | Terminal status | Process exit | Elapsed time |
|---:|---:|---|---:|---:|
| 7 | 6 | Step 6, code 10; 43 completed microsteps; 8 FGMRES iterations | 1 | 27.44 s |
| 10 | 6 | Step 6, code 10; 43 completed microsteps; 8 FGMRES iterations | 1 | 36.55 s |

The status and reported residual vector were the same. All recorded sample
columns other than GPU timing were identical, and both lip-geometry files have
the same SHA-256. Raising the cap by three iterations increased wall time by
33% without changing the accepted trajectory or moving the failure boundary.
This refutes the Newton cap as the sole cause; it does not explain or repair
the mixed-volume residual failure. No tolerance was raised, and the failed
state was not promoted to an accepted result.

## Video and limits

[![Six accepted states from the synthetic wound-lip traction run; the clip is
marked inconclusive](accepted-prefix-inconclusive-poster.png)](accepted-prefix-inconclusive.mp4)

The clip shows the six accepted 1 ms states from the v5 run at 1 fps, with no
frame interpolation. At its last accepted state, the gap is 0.594221 mm versus
the reused zero-force value of 0.600000 mm, and the force is 0.176 mN per site.
This partial change is not an endpoint, wound closure, or independent
replicate. The coupon has no measured skin fit, needle pass, suture thread,
Franka manipulation, tissue healing, or clinical qualification.

The next [stricter KKT follow-up](fgmres-followup-20261004/README.md) changes
the six-frame video and solver inputs. It records 13.06 µm of accepted gap
change versus 5.78 µm in the matched 7-Newton control, but still stops at step
6 on the same volume gate. The larger difference is solver sensitivity, not a
qualification or material-realism improvement; both clips remain inconclusive.

## Reproduction record

- Registered prediction and stop rule: [`plan.md`](plan.md).
- Bound input hashes: [`input-SHA256SUMS`](input-SHA256SUMS).
- Exact diagnostic probe source: [`source/apps/synthetic_skin_wound_traction_probe.mm`](source/apps/synthetic_skin_wound_traction_probe.mm).
- Raw control and candidate commands, exit codes, logs, per-sample CSVs, and lip
  geometry are retained under [`output/`](output/); [`SHA256SUMS`](SHA256SUMS)
  covers this record.
- Host: Apple M4 Pro Mac mini. Diagnostic checkout commit:
  `eed22b54e93db18ff8d2bf36b5c454a52efdab79`. Probe source SHA-256:
  `28f893dc2f794b2d40cac01f0545d9f14450f732b2042403d838db32ad937692`.
- The v5 run and its model limits are described in the public
  [Numi Lab media source record](https://numi-lab-research.vercel.app/media/SOURCES.md).

The follow-up residual trace and one preregistered tighter-forcing candidate
are archived in [`trace-guided-20261004/`](trace-guided-20261004/README.md).
That candidate accepted seven states and passed the unchanged `1e-4` volume
gate, but missed its stricter `5e-5` prediction; it remains a short traction
prefix, not wound closure. The per-column values are least-squares estimates,
not recomputed true residuals. These diagnostics do not support a 100×
performance claim.
