# Rigid contact response and velocity publication

## Velocity-level mechanics

For a contact between bodies `A` and `B`, with contact-point offsets `r_A` and
`r_B`, the free relative point velocity is

```math
v_c = (v_B + \omega_B\times r_B) -
      (v_A + \omega_A\times r_A).
```

The authored right-handed orthonormal frame `C=[n,u,v]` maps this into the
normal and two tangential contact axes. An additive velocity bias is expressed
in that same frame. A static-world endpoint contributes zero velocity and no
owner term.

For axis `a` and endpoint sign `s_A=-1`, `s_B=+1`, the rigid generalized
Jacobian row is

```math
J_{a,b}=s_b\begin{bmatrix}a^T & (r_b\times a)^T\end{bmatrix}.
```

With world-space inverse inertia `I_b^-1`, the generated response column is

```math
M_b^{-1}J_{a,b}^T=
s_b\begin{bmatrix}
m_b^{-1}a\\
I_b^{-1}(r_b\times a)
\end{bmatrix}.
```

The generic sparse assembler then constructs the complete shared-body
Delassus operator `W=J M^-1 J^T+R`. This is not a diagonal or independent
contact approximation: any two contacts touching the same dynamic body get a
3x3 off-diagonal block.

## Deterministic publication

After the cone solve, each body lane scans contacts in stable ascending order.
For contact impulse `lambda=(lambda_n,lambda_u,lambda_v)`, it forms

```math
p=s_b(n\lambda_n+u\lambda_u+v\lambda_v)
```

and accumulates

```math
\Delta v_b=m_b^{-1}p,
\qquad
\Delta\omega_b=I_b^{-1}(r_b\times p).
```

One SIMD32 group owns one island and one lane owns one body. There are no
floating-point atomics and no unordered reductions, so accumulation order is
fixed. The input body array is immutable; output velocities are published to a
separate range.

## Validation and transaction boundary

The response stage rejects nonfinite or out-of-range data, duplicate/static
endpoint pairs, nonpositive inverse mass, nonsymmetric or non-SPD inverse
inertia, negative cone parameters, and frames that are not right-handed and
orthonormal within the declared FP32 tolerance.

Contact spans begin invalid and commit only after the whole island validates.
Assembly begins with an invalid solver header and commits only a complete,
symmetric operator. Velocity publication requires successful response and
solver statuses. Any failure republishes the exact input body velocities.

All five stages are separate encoders on one caller-owned command buffer:

```text
rigid response -> sparse assembly -> cone solve -> rigid publication -> pose integration
```

No CPU readback, internal commit, wait, or second queue occurs between them.

## Pose integration

Successful post-contact velocity advances pose with

```math
x_{k+1}=x_k+\Delta t\,v_{k+1},
```

and the world-frame exponential quaternion update

```math
q_{k+1}=\operatorname{normalize}\left(
\begin{bmatrix}
\hat\omega\sin(\|\omega\|\Delta t/2)\\
\cos(\|\omega\|\Delta t/2)
\end{bmatrix}
\otimes q_k\right).
```

The small-angle branch uses the deterministic series for
`sin(theta/2)/theta`. Every input quaternion must already satisfy the declared
unit-norm tolerance. Any invalid pose, velocity, upstream publication, or
nonfinite result rolls the whole island back to the exact input poses.

The qualification gate also runs 240 consecutive GPU integration encoders for
one second of constant-twist free flight. It compares the final translation
and orientation against the closed-form SE(3) trajectory, checks quaternion
norm, and requires byte-identical replay.

## Present boundary

This path owns velocity-level rigid response, publication, and pose
integration. Contact topology and frames are authored inputs to the
qualification path; collision detection does not yet create or refresh them.
Gyroscopic terms, restitution policy, sleeping, continuous collision
detection, and articulated response generation are not claimed by this ABI.
