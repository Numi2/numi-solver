# Deterministic SIMD32 island solver

## Coupled problem

For an island containing `N <= 32` contacts, stack the contact impulses into

```math
\lambda \in \mathbb{R}^{3N}.
```

The island solves the convex contact-space problem

```math
\min_{\lambda\in\mathcal C}
\frac{1}{2}\lambda^T A\lambda + v_\mathrm{free}^T\lambda,
```

where `A` is the symmetric positive-definite Delassus/contact-response
operator and `C` is the product of the exact per-contact elliptic cones and
normal caps documented in [MATHEMATICS.md](MATHEMATICS.md). Because zero is
feasible, a solved island must also satisfy the energy check

```math
\frac{1}{2}\lambda^T A\lambda + v_\mathrm{free}^T\lambda \le 0
```

up to the declared FP32 tolerance.

## Packed streamed operator

The production-facing ABI stores `A` as sorted 3x3 block CSR:

- one packed header per island;
- one contiguous contact range;
- `N + 1` relative row offsets;
- strictly increasing source-contact indices in each row;
- nine row-major FP32 values per present block.

Every row must contain its diagonal, and every `A_ij` block must have a
consistent transposed partner `A_ji` within the declared symmetry tolerance.
Missing blocks are exact zeros. The kernel validates capacities, row
monotonicity, source ordering, finiteness, diagonal presence, symmetry, and
the local response factor before any physical iterate is published.

## KKT-preserving conditioning

A general 3x3 inverse followed by an ordinary Euclidean cone projection does
not, in general, preserve the minimizer of the stated convex problem. The
island solver therefore uses one positive scalar metric for all three axes of
each contact. Define

```math
d_i = \max_{a\in\{n,u,v\}}
      \sum_{j,b}|A_{(i,a),(j,b)}|,
\qquad
D = \operatorname{blockdiag}(d_0 I_3,\ldots,d_{N-1}I_3).
```

For symmetric positive-definite `A`, `D - A` is symmetric diagonally dominant
with nonnegative diagonal and is therefore positive semidefinite. Hence

```math
0 \prec D^{-1/2} A D^{-1/2} \preceq I.
```

The metric is scalar within each cone block, so projection in the `D` metric
is exactly the existing Euclidean elliptic-cone projection. Conditioning thus
bounds the parallel step without changing its KKT fixed point.

## Deterministic iteration and residual

One SIMD32 group owns one island and one lane owns one contact. At iteration
`k`, every lane reads the same immutable impulse generation and evaluates

```math
r^k = A\lambda^k + v_\mathrm{free},
```

```math
p^k = \Pi_{\mathcal C}
      \left(\lambda^k-D^{-1}r^k\right),
```

```math
\lambda^{k+1}=(1-\omega)\lambda^k+\omega p^k,
\qquad 0 < \omega \le 1.
```

The first 16 iterations retain this unaccelerated map, preserving the common
short-island path. When `omega=1` and the island remains unresolved, the
solver switches to the same `D`-metric accelerated proximal map:

```math
y^k=\lambda^k+\beta_k(\lambda^k-\lambda^{k-1}),
```

```math
\lambda^{k+1}=\Pi_{\mathcal C}
\left(y^k-D^{-1}(Ay^k+v_\mathrm{free})\right),
```

```math
t_{k+1}=\frac{1+\sqrt{1+4t_k^2}}{2},
\qquad
\beta_{k+1}=\frac{t_k-1}{t_{k+1}}.
```

Acceleration is disabled for under-relaxed inputs. The whole SIMD group
restarts momentum when

```math
\sum_i (y_i^k-\lambda_i^{k+1})^T
(\lambda_i^{k+1}-\lambda_i^k)>0,
```

when an extrapolated point first reaches the provisional tolerance, or after
64 iterations without an earlier restart. The first condition detects
momentum that is no longer aligned with proximal progress. The second forces
the next certificate to evaluate the accepted feasible iterate with
`beta=0`; an extrapolated search point never terminates the solve. The fixed
restart bounds momentum when the adaptive test remains silent.

Rows visit source contacts in strictly increasing order. SIMD reductions have
fixed topology. No floating-point atomics, unordered append queues,
host-visible intermediate counts, or lane-0 serial island solve enters the
physical iteration.

The convergence certificate is the scaled KKT gradient mapping

```math
G_D(\lambda)=D\left(
\lambda-\Pi_{\mathcal C}(\lambda-D^{-1}r(\lambda))
\right),
```

normalized as

```math
R_\mathrm{KKT}=
\frac{\|G_D(\lambda)\|_\infty}
{\max(1,\|v_\mathrm{free}\|_\infty,\|A\lambda\|_\infty)}.
```

Unlike an unscaled update norm, this certificate cannot become artificially
small merely because an operator row has a small step size. The candidate is
not applied after the current iterate satisfies the gate, so the published
impulses, objective, and residual all refer to the same vector.

The status diagnostic records the number of accelerated restarts. Easy
islands that finish before momentum begins report zero.

## Transaction and failure behavior

The kernel rejects invalid ABI/ranges, malformed or asymmetric CSR,
nonfinite input, failed local conditioning, nonfinite arithmetic, and bounded
iteration exhaustion with typed status. A valid but unconverged island
republishes its projected warm-start checkpoint. Invalid or nonfinite initial
state publishes zero. No partial iterate is exposed as a solved result.

The dense kernel remains an executable qualification oracle. The harness
constructs the same operator in dense and streamed form and requires
byte-identical FP32 impulses, statuses, KKT residuals, objectives, iteration
counts, and rollback behavior. An independent FP64 solve, Cholesky SPD check,
cone feasibility, nonpositive objective check, and the analytic two-contact
shared-rigid solution `(1/3, 1/3)` provide separate mathematical evidence.
