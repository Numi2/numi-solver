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

## Implicit contact law

The rigid ABI derives its velocity target and diagonal conditioning from a
versioned material law rather than accepting a host-chosen matrix. For contact
axis `a`, timestep `h`, stiffness `k_a`, and damping `d_a`, it uses the
implicit spring-damper denominator and constraint-force mixing term

```math
s_a=d_a+h k_a,
\qquad
\gamma_a=\frac{1}{h s_a}.
```

All three denominators must be finite and strictly positive. The generated
regularization is

```math
R_i=\operatorname{diag}(\gamma_n,\gamma_u,\gamma_v),
```

which is written on GPU and consumed by sparse assembly in the following
encoder.

For signed gap `g`, penetration slop `epsilon`, and optional maximum recovery
speed `v_max`, the normal recovery target is

```math
c=\min(g+\epsilon,0),
\qquad
v_\mathrm{recovery}=
\min\left(v_\max,-\frac{k_n c}{s_n}\right).
```

A zero `v_max` means uncapped recovery. For pre-solve normal velocity `v_n`,
restitution `e` and impact threshold `v_threshold`, the impact target is

```math
v_\mathrm{impact}=
\begin{cases}
-e v_n,&v_n<-v_\mathrm{threshold},\\
0,&\text{otherwise}.
\end{cases}
```

The solver receives

```math
v_{\mathrm{free},n}=v_n-
\max(v_\mathrm{recovery},v_\mathrm{impact})+b_n,
```

while the tangential axes retain their authored additive bias. Thus impact and
penetration recovery share one unilateral normal target instead of stacking
two impulses.

The physical energy gate accounts for deliberate stabilization work. If
`W_p=J M^-1 J^T`, `Gamma=R`, and `v_target` contains only the normal targets,
the accepted impulse must satisfy

```math
\Delta K
=\lambda^T v_\mathrm{raw}+\frac12\lambda^T W_p\lambda
\le
\lambda^T v_\mathrm{target}-\frac12\lambda^T\Gamma\lambda.
```

The qualification harness checks this bound from measured before/after body
energy and independently reconstructed targets; it does not mislabel bounded
penetration-recovery work as numerical energy creation.

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
inertia, negative cone parameters, invalid stiffness/damping/timestep,
restitution outside `[0,1]`, and frames that are not right-handed and
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
