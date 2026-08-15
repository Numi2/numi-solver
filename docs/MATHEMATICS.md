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
\widehat A = \frac{A + A^T}{2s} + 0.01 I,
\qquad
P = (s\widehat A)^{-1}.
```

It fails closed for nonfinite input, `s <= 1e-10`, or a nonpositive/nonfinite
regularized determinant. The `0.01 I` term is a deterministic CFM floor: it
bounds amplification from a weak response mode but introduces deliberate
compliance in that mode.

## Projected update

The isolated probe executes the same local update used by the production
solver:

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
tangent point on the capped ellipse. A zero friction axis retains the prior
ABI behavior and disables both tangential impulses.

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
- normalized fixed-point convergence residual;
- byte-identical repeated GPU output;
- separating, impact, sticking, sliding, anisotropic, ill-conditioned and
  capped-contact, polar-boundary and extreme-anisotropy cases;
- isolated kernel time and problem throughput.

This qualifies the local Temporal Cone block. It does not yet qualify contact
generation, multi-contact island coupling, velocity publication, integration,
or a complete physical trajectory.
