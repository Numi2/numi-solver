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

## Contact law and publication

The articulated path uses the same velocity-level spring/damper,
penetration-recovery, restitution, and elliptic-friction contract as the rigid
path. The free contact velocity is derived on GPU from `J_c v`. After a
successful cone solve, one lane owns each generalized coordinate and scans
contacts in canonical order:

```math
v^+=v+\sum_i M^{-1}J_{c,i}^T\lambda_i.
```

If operator validation, factor response, assembly, convergence, or finite
publication fails, the complete input generalized-velocity vector is
republished unchanged.

## Independent evidence

The qualification mechanism is a fixed-base planar two-link chain with two
simultaneous articulated contacts. A separate FP64 implementation derives its
center-of-mass Jacobians, mass matrix from translational and rotational
kinetic energy, point Jacobians, factor solves, and energy budget. Central
finite differences independently check the analytic point Jacobians over a
wide joint-angle sweep. This proves the declared mechanism and numerical path;
it is not evidence for collision detection, arbitrary robot import,
articulated position integration, or complete trajectories.
