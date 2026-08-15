# Temporal Cone mathematics

## Local contact problem

For one contact, Temporal Cone solves for the impulse

```math
\lambda = (\lambda_n, \lambda_u, \lambda_v)^T
```

in the normal and two tangent directions. The local point response is

```math
A = J M^{-1} J^T + R,
```

where `J` is the contact Jacobian, `M` is the rigid/articulated mass operator,
and `R` is authored row regularization. Given free relative velocity
`v_free`, the local residual is

```math
r(\lambda) = v_\mathrm{free} + A\lambda.
```

The unilateral and elliptic Coulomb bounds are

```math
0 \le \lambda_n \le \lambda_{n,\max},
\qquad
\left(\frac{\lambda_u}{\mu_u\lambda_n}\right)^2 +
\left(\frac{\lambda_v}{\mu_v\lambda_n}\right)^2 \le 1.
```

A zero maximum-normal value means that no upper cap is authored.

## Deterministic conditioning

The FP32 point response may be redundant or nearly rank deficient. The live
Metal path computes

```math
s = \max_{i,j}|A_{ij}|,
\qquad
S = \frac{A + A^T}{2s},
\qquad
\widehat A = S + 0.01 I,
\qquad
P = (s\widehat A)^{-1}.
```

Before applying the shift, all principal minors of `S` must certify
`S` as positive-semidefinite within the declared FP32 tolerance. This
distinguishes physical rank
deficiency from negative curvature: a positive determinant with two negative
eigenvalues is rejected, and the CFM floor cannot hide it. The shifted matrix
must then pass the strict Sylvester SPD test with a finite representable
inverse scale. The `0.01 I` term bounds amplification from a weak response
mode but introduces deliberate compliance in that mode.

## Isolated conditioned map

The isolated probe executes the imported local conditioned map:

```math
\lambda^{k+1} =
\Pi_{\mathcal C}\left(\lambda^k - P r(\lambda^k)\right).
```

`Pi_C` is the Euclidean closest-point projection onto the complete cone. For
isotropic friction `mu`, the boundary projection is closed form. With
`r = hypot(y_u, y_v)`:

```math
\lambda_n = \frac{y_n + \mu r}{1 + \mu^2},
\qquad
(\lambda_u,\lambda_v) =
\frac{\mu\lambda_n}{r}(y_u,y_v).
```

Inputs already inside the cone remain unchanged. Inputs in its polar cone map
to zero. This fast path contains no iteration.

For anisotropic friction, the closest point is found through the single
monotone KKT multiplier `tau`. With `lambda_n = y_n + tau`, the boundary root
is

```math
\sum_{i\in\{u,v\}}
\frac{y_i^2\mu_i^2}
     {(\mu_i^2(y_n+\tau)+\tau)^2}=1.
```

Metal brackets this root and executes 28 ordered bisection iterations. The
FP64 oracle converges the same mathematical root to double precision rather
than copying the FP32 iteration count. If the unbounded projection exceeds an
authored normal cap, a second monotone scalar projection finds the closest
tangent point on the capped ellipse.

For finite extreme-magnitude inputs, direct FP32 squared norms can overflow
even when the exact projected impulse is representable. Cone projection is
positively homogeneous when the impulse and normal cap are scaled together.
With

```math
s=\max\left(\|y\|_\infty,\lambda_{n,\max}\right),
```

the solver therefore uses

```math
\Pi_{\mathcal C(\lambda_{n,\max})}(y)
=s\,\Pi_{\mathcal C(\lambda_{n,\max}/s)}(y/s).
```

This normalization is selected only when `s > 1e18`; the ordinary path keeps
the original closed-form or scalar-root projection. A finite input whose exact
projected result exceeds FP32 range is rejected by the existing typed,
transactional nonfinite-result gate rather than silently clamped.

A single zero friction coefficient defines a lower-dimensional cone rather
than a frictionless contact. For example, when `mu_u=0` and `mu_v>0`,

```math
\lambda_u=0,\qquad |\lambda_v|\le\mu_v\lambda_n.
```

The inactive tangent is projected exactly to zero. The remaining normal and
active tangent form a two-dimensional Coulomb wedge with the closed-form
boundary projection

```math
\lambda_n=\frac{y_n+\mu|y_t|}{1+\mu^2},\qquad
\lambda_t=\operatorname{sign}(y_t)\mu\lambda_n.
```

This avoids the anisotropic bisection loop. At an authored normal cap, the
active tangent is clamped directly to

```math
[-\mu\lambda_{n,\max},\,+\mu\lambda_{n,\max}].
```

Only `mu_u=mu_v=0` disables both tangential impulses.

The probe qualifies this map as an FP32 numerical building block. A general
3x3 `P` followed by Euclidean projection is not, by itself, a KKT certificate
for the quadratic objective with Hessian `A`. The coupled island solver avoids
that mismatch by using a scalar metric on each complete cone block; see
[ISLAND_SOLVER.md](ISLAND_SOLVER.md).

## Convergence residual

The local fixed-point residual is the final projected update:

```math
R_\mathrm{fp} =
\frac{\|\lambda^{k+1}-\lambda^k\|_\infty}
     {\max(1,\|\lambda^{k+1}\|_\infty)}.
```

The qualification default is 16 iterations and requires
`R_fp <= 2e-6`. In the adversarial Apple M4 sweep, eight iterations left a
`2.34e-3` absolute edge-case update; 16 reduced the normalized maximum below
`4.8e-7`. At 32 and 64 iterations the observed floor did not improve, so 16
is the current accuracy/throughput point rather than an arbitrary larger
budget.

## What the harness measures

`numi-solver-math` runs deterministic problem batches through the real Apple
Metal kernel and compares every output with an independent FP64 CPU
implementation of the equations above. It checks:

- FP32-versus-FP64 impulse and residual error;
- finite, nonnegative and capped normal impulses;
- exact elliptic-cone projection against the FP64 closest point;
- exact one-axis degenerate cones and their capped interval projection;
- normalized fixed-point convergence residual;
- byte-identical repeated GPU output;
- separating, impact, sticking, sliding, anisotropic, ill-conditioned and
  capped-contact, polar-boundary, extreme-anisotropy, and finite `1e20`-scale
  isotropic, anisotropic, and capped projection cases;
- isolated kernel time and problem throughput.

This qualifies the local Temporal Cone block. Coupled contact-space KKT,
streamed-operator, SPD, objective, and deterministic-island evidence is
reported separately in [QUALIFICATION.md](QUALIFICATION.md). Collision
generation, rigid/articulated velocity publication, integration, and a
complete physical trajectory remain separate qualification layers.
