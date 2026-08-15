# Articulated contact mechanics

## Owning operator

`ArticulatedOperator.metal` is the canonical numerical owner imported from the
recorded `numisolver` revision. Given an immutable tree and configuration `q`,
it computes body poses, analytic world-point Jacobians, and the generalized
mass matrix. It adds authored armature inertia, rejects non-SPD or inaccurate
systems, and publishes the checked lower Cholesky factor `L` such that

```math
M(q)=L L^T.
```

Long body-order mass reductions and Cholesky inner products use deterministic
Neumaier compensation. This reduces loss of distal inertia contributions
without changing storage precision or pretending FP32 has FP64 range.

The standalone adapter supports at most 32 generalized velocity coordinates
per connected articulation so each contact term remains compatible with the
generic assembly ABI. The imported owner has a wider 64-DoF operator capacity;
the narrower contact adapter fails explicitly rather than truncating.

## Contact response

For a queried body point, the owner publishes

```math
J_p(q)\in\mathbb{R}^{3\times n_v}.
```

With right-handed contact frame `F=[n,t_u,t_v]`, the adapter forms

```math
J_c=F^T J_p,
\qquad
L L^T X=J_c^T,
\qquad
X=M^{-1}J_c^T.
```

Each active SIMD32 contact lane performs three deterministic forward/backward
triangular solves. It evaluates a normalized backward error before publishing
the term. No explicit inverse, atomics, unordered reduction, or host solve is
used. Contacts on the same articulation share one stable owner ID, so the
generic assembler creates every required off-diagonal block

```math
W_{ij}=J_{c,i}M^{-1}J_{c,j}^T+\delta_{ij}\Gamma_i.
```

## Conditioning admission

A small backward residual only proves accuracy for the assembled FP32 matrix;
it does not guarantee a small forward error relative to the intended physical
mass matrix. Before any response column is exposed, DoF lane `i` therefore
reconstructs row `i` of `M` and solves `M x=e_i`. Symmetry makes the resulting
inverse column's absolute sum equal the corresponding inverse-row sum. The
SIMD group computes

```math
\kappa_\infty(M)=\lVert M\rVert_\infty
                  \lVert M^{-1}\rVert_\infty.
```

The dense mode of ABI v3 admits at most `16384`. A larger or nonfinite estimate publishes typed
`CONDITIONING_FAILED` status and preserves the complete input generalized
velocity. The threshold does not silently modify mass. Better conditioning
must come from explicit physical armature, implicit drive inertia, a better
operator, or a caller decision.

## Streamed inverse-ABA candidate

The repository also carries the owning articulated-body inverse-mass action.
For each environment it factorizes spatial articulated inertias once, then
streams every contact-axis right-hand side through deterministic reverse and
forward tree sweeps:

```math
J_{c,i}^{T}\longmapsto M(q)^{-1}J_{c,i}^{T}.
```

A separate GPU preparation kernel consumes the kinematics-only point
Jacobians and rotates them into the same `J_c` rows as the factor-backed path.
At 32 DoFs both the kinematics-only Jacobian and prepared contact Jacobian are
bit-identical to their dense-path counterparts. The inverse action does not
form `M`, `L`, or `M^{-1}` and uses no host solve.

ABI v3 routes this path through a complete seven-stage candidate transaction:
kinematics, contact-Jacobian preparation, inverse ABA, response finalization,
sparse assembly, cone solve, and generalized-velocity publication. All stages
execute on one command buffer without CPU readback. Finalization transposes the
streamed columns into the generic assembly layout only after checking the
preparation and inverse statuses, finite values, capacities, and contact laws;
later failures preserve the complete input velocity.

This is still a candidate, not the default. Its response columns, assembled
Delassus blocks, solve, publication, rollback, and energy budget are checked
against independent definitions, but its squared articulated-body pivot ratio
is only a diagnostic. It is not a forward-error condition estimate comparable
to `kappa_infinity(M)`. The factor-backed path therefore retains production
ownership until the inverse path has a state-local admission gate with a
declared physical meaning.

## Contact law and publication

The articulated path uses the same velocity-level spring/damper,
penetration-recovery, restitution, and elliptic-friction contract as the rigid
path. The free contact velocity is derived on GPU from `J_c v`. After a
successful cone solve, one lane owns each generalized coordinate and scans
contacts in canonical order:

```math
v^+=v+\sum_i M^{-1}J_{c,i}^T\lambda_i.
```

If operator validation, dense-factor or inverse response, assembly,
convergence, or finite publication fails, the complete input
generalized-velocity vector is republished unchanged.

## Independent evidence

The qualification family contains fixed-base planar serial chains from two to
32 DoFs, with one simultaneous contact at every distal link endpoint. A
separate FP64 implementation derives each center-of-mass Jacobian, mass matrix
from translational and rotational kinetic energy, exact infinity condition,
point Jacobian, factor solve, Delassus block, and energy budget. Central finite
differences independently check analytic point Jacobians. The 32-DoF capacity
case uses explicitly authored armature sufficient to pass ABI conditioning;
the same mechanism with low armature is separately required to fail and roll
back. This proves the declared mechanisms and numerical path; it is not
evidence for collision detection, arbitrary robot import, articulated
position integration, or complete trajectories.
