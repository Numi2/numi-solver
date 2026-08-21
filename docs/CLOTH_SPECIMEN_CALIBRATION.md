# Cloth specimen calibration

This protocol converts the remaining cloth-realism boundary into measured,
held-out evidence. It does not claim that the current produce-bag constants are
already calibrated. The committed CSV files contain parameter defaults,
schema, and synthetic test data only; they contain no physical specimen data.

## Evidence contract

A physical calibration campaign must preserve:

- a specimen identifier, unloaded mass, temperature, humidity, conditioning
  history, and photographs;
- raw sensor files, units, device calibration records, sample rate, timestamps,
  and cryptographic hashes;
- measured unloaded geometry and the full six-degree-of-freedom grip trajectory;
- a trial-level split declared before fitting; frames from one trial may not be
  divided between calibration and held-out sets;
- the solver revision, parameter baseline, perturbation size, executable
  arguments, state hashes, and failure flags used to construct each prediction;
- measured uncertainty `sigma` for every observable, estimated from instrument
  accuracy and repeated trials rather than chosen to make residuals pass.

Simulation, rendering, or synthetic CSV data cannot substitute for these
records.

## Measurements

Use at least three specimens from the same product lot and repeat each static
test at least five times. Reserve at least one whole specimen and one whole
trajectory of each dynamic trial type for held-out evaluation.

| Family | Required measurement | Solver quantities informed |
|---|---|---|
| Geometry | Photogrammetry or structured-light scan of the unloaded wall, folded cuff, mouth, and closed bottom; knot spacing and yarn diameter at 20 locations | authored rest positions, yarn radius, topology scale |
| Mass | Whole-bag mass plus separated cuff/body mass when destructive sampling is allowed | ordinary and hem node mass |
| Axial | Warp, weft, bottom, and cuff strip force-extension cycles at three loads and three rates, including unloading | body/cuff axial compliance, rate dependence not represented by the present model |
| Crossing | Local warp/weft angle versus applied moment and load, including knot slip threshold | knot compliance and any future slip law |
| Bend | Cantilever or loop-curvature tests for body and cuff yarn paths | body/cuff bend compliance |
| Friction | Pull or incline tests for cloth/plane, cloth/cloth, fruit/cloth, fruit/plane, and fruit/fruit, with measured normal load | five Coulomb coefficients |
| Rolling | Fruit deceleration and angular velocity on the measured plane | fruit rolling resistance |
| Air | Yarn-strip and fruit force/torque versus relative air speed and orientation | yarn and fruit drag coefficients |
| Grounded | Synchronized 3D knot/fruit tracks after release onto the plane | combined mass, compliance, contact, friction, and damping outcome |
| Pickup | Six-axis seam load, grip pose, mouth shape, fruit tracks, exit time, and landing state | grip compliance, cuff mechanics, release, contact, and trajectory outcome |
| Swing | Six-axis seam load and pose plus 3D wall/fruit tracks over several orbit radii and rates | coupled compliance, bend, air load, containment, and lag |
| Spill | Mouth pose, per-fruit exit classification, impact time, and final state | held-out release and ground-contact outcome |

If repeat data show systematic rate dependence, hysteresis, plastic set, creep,
damage, moisture dependence, or knot sliding, fitting the present constants is
not sufficient. The owning constitutive law must be extended and requalified.

## Linearized fit

The calibration executable fits multiplicative parameter changes. For a
positive parameter \(\theta_j\), define

```math
x_j = \log(\theta_j / \theta_{j,0}).
```

For each measured observable \(y_i\), generate a baseline prediction and a
central finite-difference sensitivity using an exact replay of the measured
trial:

```math
S_{ij} \approx
\frac{y_i(\theta_{j,0} e^h)-y_i(\theta_{j,0} e^{-h})}{2h},
\qquad h=0.05.
```

The local prediction is

```math
\widehat y_i = y_{i,0} + \sum_j S_{ij}x_j.
```

Only rows marked `calibration` enter the weighted solve. Rows marked `heldout`
are evaluated afterward and never influence the parameter estimate. Weights
are \(1/\sigma_i^2\). Sensitivity columns are normalized before solving; the
unregularized normalized design matrix must retain full rank. A tiny optional
ridge term stabilizes the solve but cannot make a rank-deficient experiment
qualify.

Run the synthetic recovery example:

```sh
./build/numi-solver-cloth-calibration \
  --parameters calibration/synthetic/parameters.csv \
  --observations calibration/synthetic/qualified-observations.csv \
  --material-output build/synthetic-cloth-material.txt
```

For real data, copy
[`calibration/observations-template.csv`](../calibration/observations-template.csv),
fill every measured/baseline/sensitivity field, and use
[`calibration/cloth-parameters-v1.csv`](../calibration/cloth-parameters-v1.csv)
as the current parameter contract.

The default acceptance gates are:

- full sensitivity rank and normalized condition number at most `1e4`;
- every fitted parameter inside its declared physical bounds;
- calibration weighted RMSE at most `2 sigma`;
- held-out weighted RMSE at most `2 sigma` and worst observation at most
  `4 sigma`;
- held-out RMSE strictly better than the authored baseline.

The executable exits `1` on a failed qualification and refuses to write a
material artifact from a failed fit. It fingerprints both input CSV files in
the report and artifact.

## Ownership map

The parameter names correspond to live CPU/Metal host values as follows:

| Calibration name | Current owner |
|---|---|
| `ordinary_node_mass_kg`, `hem_node_mass_kg` | particle construction in `tools/cloth_bag.cpp` and `tools/cloth_bag_metal.mm` |
| `yarn_radius_m` | contact radius and `clothMaterial.x` |
| `axial_body_compliance_m_per_n`, `axial_cuff_compliance_m_per_n` | axial distance constraints |
| `knot_compliance` | warp/weft crossing-angle constraints |
| `bend_body_compliance_m_per_n`, `bend_cuff_compliance_m_per_n` | three-knot bend chords |
| `grip_compliance_m_per_n` | ten-knot top-cuff attachment |
| friction and rolling names | cloth/fruit contact material fields |
| aerodynamic names | `airVelocityAndDensity` and `aerodynamicCoefficients` |

Both cloth executables consume a complete
`numi.cloth.material.v1` artifact through `--material FILE`. The shared parser
rejects unknown, duplicate, missing, non-finite, and out-of-contract fields and
requires both input fingerprints. CPU and Metal parity tests load an explicit
synthetic artifact containing the authored defaults and require exactly the
same published physical-state hashes as the no-artifact path. This proves that
the runtime handoff does not silently change the authored baseline; it does not
calibrate that baseline.

The runtime reports `material_artifact_loaded=true`, not `calibrated=true`.
Loading proves schema, bounds, and provenance-field transport only. The
calibration executable's separate `qualified=true` report is the fit gate, and
the hashes identify inputs without proving that those inputs came from a real
specimen.

Only a fit using the complete 19-parameter contract writes an artifact that the
cloth runtimes can consume; the smaller synthetic recovery system intentionally
does not. No real fitted artifact is committed yet. Unloaded geometry and grip
motion remain trajectory inputs, not material-fit parameters, and must be
replaced by the measured scan and recorded six-degree-of-freedom seam pose.
The latter now has a strict executable format documented in
[GRIP_TRAJECTORY.md](GRIP_TRAJECTORY.md).

For example, after a real fit qualifies:

```sh
./build/numi-solver-cloth-bag \
  --material build/measured-cloth-material.txt \
  --scenario pickup --steps 240 --substeps 48 --iterations 32

./build/numi-solver-cloth-metal \
  --material build/measured-cloth-material.txt \
  --replays 2 --iterations 32 --strain-sweeps 3 \
  --pickup-prefix build/measured-pickup --pickup-steps 480
```

## Synthetic rejection coverage

CTest covers four mathematical boundaries:

- exact recovery and untouched held-out prediction for an identifiable
  three-parameter system;
- rejection of two indistinguishable sensitivity columns;
- rejection of a fit that explains calibration rows but fails held-out
  trajectories;
- rejection when any trial identifier appears in both calibration and held-out
  rows, preventing frame or observable leakage from one physical trial.

These tests validate the fitting and leakage boundaries only. They are not
physical evidence for the produce bag. A separate synthetic pipeline test
builds a full-rank 19-parameter identity experiment, writes the qualified
artifact, and requires both CPU and Metal runtimes to parse its exact
provenance and pass their physics checks. It validates the
calibration-to-runtime connection, not the authored parameter values.
