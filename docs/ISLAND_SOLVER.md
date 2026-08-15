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

where `A` is the symmetric positive-semidefinite Delassus/contact-response
operator (positive-definite when regularization removes every null mode) and
`C` is the product of the exact per-contact elliptic cones,
lower-dimensional one-axis cones, and normal caps documented in
[MATHEMATICS.md](MATHEMATICS.md). Because zero is
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

For symmetric positive-semidefinite `A`, `D - A` is symmetric diagonally dominant
with nonnegative diagonal and is therefore positive semidefinite. Hence

```math
0 \preceq D^{-1/2} A D^{-1/2} \preceq I.
```

The metric is scalar within each cone block, so projection in the `D` metric
is exactly the existing Euclidean elliptic-cone projection. Conditioning thus
bounds the parallel step without changing its KKT fixed point.

Each unshifted 3x3 diagonal response block is independently admitted as PSD
from all principal minors. The island path does not compute or use a local
inverse; this certificate therefore adds no fictitious regularization to the
operator being iterated.

PSD also requires every cross-contact scalar coupling to obey the
Cauchy-Schwarz principal-minor bound

```math
|A_{ij}|^2\le A_{ii}A_{jj}.
```

Dense and streamed kernels check each unordered cross-contact pair exactly
once before iteration, with the three source axes checked as one vector. They
form `sqrt(A_ii) sqrt(A_jj)` from separately rooted diagonals, avoiding the
overflow-prone product `A_ii A_jj`; the comparison tolerance is relative to
the larger of that bound and `|A_ij|`. The streamed path temporarily stores
the diagonal roots in the rollback checkpoint arena, adding no retained
threadgroup memory.

This is a necessary global-curvature condition, not a claim that checking only
2x2 minors proves an arbitrary matrix PSD. The production assembly path adds a
complete packed FP32 semidefinite-Cholesky certificate before it publishes a
solver header. Direct dense/CSR callers still retain the final whole-island
positive-energy certificate, which rejects a feasible stationary point with
positive energy but is not a substitute for full arbitrary-matrix PSD proof.

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

Success requires three simultaneous finite certificates. With

```math
t_\mathrm{KKT}=t_\mathrm{abs}+t_\mathrm{rel}
\max(1,\|v_\mathrm{free}\|_\infty,\|A\lambda\|_\infty),
```

```math
t_\mathcal C=t_\mathrm{abs}+t_\mathrm{rel}
\max(1,\|\lambda\|_\infty),
\qquad
t_E=3N\,t_\mathrm{KKT}\|\lambda\|_\infty,
```

the kernel admits the candidate only when

```math
\|G_D(\lambda)\|_\infty\le t_\mathrm{KKT},
\qquad
V_\mathcal C(\lambda)\le t_\mathcal C,
\qquad
E(\lambda)\le t_E.
```

The last tolerance bounds the FP32 first-order energy uncertainty across
`3N` impulse components. It does not turn a positive-energy stationary point
into a solved contact state. Any nonfinite row bound, reciprocal step,
iteration diagnostic, reduced residual, feasibility value, impulse scale, or
objective is a typed arithmetic failure rather than a false convergence.

The status diagnostic records the number of accelerated restarts. Easy
islands that finish before momentum begins report zero.

The final feasibility diagnostic evaluates normal nonnegativity, the authored
normal cap, every positive-friction normalized axis, and every zero-friction
axis separately. A valid impulse on a one-axis cone is therefore not mistaken
for frictionless contact, while any impulse on its inactive tangent is a
certificate failure.

The feasibility value has impulse units, matching `t_C`. For each active
tangent define `q_i=lambda_i/mu_i`, and set an inactive component of `q` to
zero while separately admitting `|lambda_i|` as a violation. The certificate
is

```math
V_\mathcal C(\lambda)=\max\left(
-\lambda_n,
\lambda_n-\lambda_{n,\max},
\|q\|_2-\max(\lambda_n,0),
\max_{i:\mu_i=0}|\lambda_i|,
0
\right),
```

with the cap term omitted when the cone is unbounded. The two-dimensional
norm uses max-component scaling before squaring. This prevents overflow and,
unlike a dimensionless radius ratio, cannot become easier to satisfy merely
because the impulse magnitude makes the relative tolerance larger.

## Transaction and failure behavior

The kernel rejects invalid ABI/ranges, malformed or asymmetric CSR,
nonfinite input, failed local or cross-contact curvature admission,
nonfinite derived row scaling,
nonfinite iteration/final arithmetic, failed KKT/feasibility/energy admission,
and bounded iteration exhaustion with typed status. A valid but uncertified
island republishes its projected warm-start checkpoint. Invalid or nonfinite
initial state publishes zero. No partial iterate is exposed as a solved
result.

The dense kernel remains an executable qualification oracle. The harness
constructs the same operator in dense and streamed form and requires
byte-identical FP32 impulses, statuses, KKT residuals, objectives, iteration
counts, and rollback behavior. An independent FP64 solve, Cholesky SPD check,
cone feasibility, nonpositive objective check, and the analytic two-contact
shared-rigid solution `(1/3, 1/3)` provide separate mathematical evidence.
