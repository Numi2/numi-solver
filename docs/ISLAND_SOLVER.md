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
normal caps documented in [MATHEMATICS.md](MATHEMATICS.md).

## Block conditioning and stability

Each contact owns the conditioned inverse `P_i` of its local 3x3 diagonal
block. The complete block-Jacobi preconditioner is

```math
P = \operatorname{blockdiag}(P_0,\ldots,P_{N-1}).
```

The GPU computes a deterministic absolute-row-sum bound

```math
L_\infty = \|PA\|_\infty
```

and limits the authored relaxation to

```math
\omega = \min\left(\omega_\mathrm{authored},
                    \frac{1}{\max(L_\infty,1)}\right).
```

This prevents strongly shared contact modes from destabilizing the parallel
update. It is derived entirely from the immutable operator; there is no
host-tuned per-scene branch or residual-dependent floating-point race.

## Deterministic iteration

One SIMD32 group owns one island and one lane owns one contact. At iteration
`k`, every lane reads the same immutable impulse generation and evaluates

```math
r^k = A\lambda^k + v_\mathrm{free},
```

```math
d^k =
\Pi_{\mathcal C}(\lambda^k-Pr^k)-\lambda^k,
```

```math
\lambda^{k+1}=\lambda^k+\omega d^k.
```

All matrix rows visit source contacts in ascending stable order. SIMD maximum
and sum reductions have fixed topology. No floating-point atomics, unordered
append queues, host-visible intermediate counts, or lane-0 serial island solve
enters the iteration.

The natural convergence certificate is

```math
R_\mathrm{natural}=
\frac{\|d^k\|_\infty}{\max(1,\|\lambda^k\|_\infty)}.
```

The candidate is not applied after its current iterate satisfies the gate.
The published impulses and convergence certificate therefore refer to the
same vector.

## Transaction and failure behavior

The kernel rejects nonfinite or nonsymmetric operators, invalid contracts,
and failed local conditioning with typed status. Iteration exhaustion returns
`NUMI_TEMPORAL_CONE_ISLAND_DID_NOT_CONVERGE`. Any rejected island republishes
its projected warm-start checkpoint instead of exposing a partial iterate.

The standalone dense matrix is a qualification representation. It makes every
cross-contact coefficient directly inspectable and supports an independent
FP64 oracle. Production sparse/streamed operators may replace storage, but
must preserve the same matrix action, stable summation order, cone map,
convergence certificate, and rollback contract.
