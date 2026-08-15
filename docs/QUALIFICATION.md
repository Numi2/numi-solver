# Temporal Cone qualification

## Apple M4 local-block gate

Measured on 2026-08-15 with Apple Metal toolchain `32023.883`:

```sh
./build/numi-solver-math --cases 65536 --replays 5 --iterations 16
./build/numi-solver-math --cases 65536 --replays 5 --iterations 16 --isotropic
```

The qualification gate requires:

- zero unexpected case failures;
- maximum FP32-versus-FP64 relative error at most `1e-5`;
- maximum elliptic-cone violation at most `2e-6`;
- maximum normalized fixed-point residual at most `2e-6`;
- byte-identical outputs across every replay;
- separating contact maps to zero;
- the isotropic sliding case lies on the cone;
- the authored normal cap is respected.

The adversarial anisotropic batch produced:

```text
cases=65536
replays=5
solver_iterations=16
failed_cases=0
max_fp64_relative_error=0.000003368
max_cone_violation=0.000000119
max_relative_fixed_point_residual=0.000000477
deterministic_replay=true
```

The measured isolated local-block throughput was approximately 33.4 million
problems/s for the adversarial anisotropic batch and 103.4 million problems/s
for the predominantly isotropic batch. The latter takes the closed-form fast
path; these are local mathematical blocks, not full environment steps.

The resulting metallib SHA-256 was
`cabe8df70c136642750bccd8534cf6f975282fe9ef90b5c32cc289d77f8a03b0`.

## Evidence boundary

These measurements execute the real conditioned inverse and cone projection
helpers from `MetalWorldContact.metal` on Apple GPU. The FP64 implementation is
independent and solves anisotropic/capped scalar roots to double precision.

This is strong evidence for local contact mathematics, FP32 stability,
determinism and isolated kernel cost. It is not evidence for collision
generation, coupled multi-contact island convergence, rigid/articulated
velocity publication, integration, or a complete physical trajectory. Those
remain separate qualification layers.
