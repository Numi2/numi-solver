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

`Pi_C` first clamps the normal impulse, then radially scales the two tangent
components onto the authored ellipse when they exceed it. This is exact
enforcement of the implemented friction bound for the accepted normal
impulse; it is not claimed to be the Euclidean closest-point projection onto
the complete three-dimensional cone.

## What the harness measures

`numi-solver-math` runs deterministic problem batches through the real Apple
Metal kernel and compares every output with an independent FP64 CPU
implementation of the equations above. It checks:

- FP32-versus-FP64 impulse and residual error;
- finite, nonnegative and capped normal impulses;
- elliptic friction-bound feasibility;
- byte-identical repeated GPU output;
- separating, impact, sticking, sliding, anisotropic, ill-conditioned and
  capped-contact cases;
- isolated kernel time and problem throughput.

This qualifies the local Temporal Cone block. It does not yet qualify contact
generation, multi-contact island coupling, velocity publication, integration,
or a complete physical trajectory.
