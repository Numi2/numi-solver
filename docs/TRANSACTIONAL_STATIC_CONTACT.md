# Transactional static-contact subdivision

The CPU host driver now completes a requested interval only after every child
step is accepted. Rejected attempts use private, owned copies of physical,
constraint warm and controller state. A later fatal failure, exhausted budget
or unrepresentable half timestep returns the original whole interval and
commits zero time. Successful children receive the exact authoritative
controller sample at their own interval endpoint.

[Source and binary receipt](assets/transactional-substep-evidence.json) ·
[Host result](assets/transactional-substep-host.log) ·
[CMake result](assets/transactional-substep-ctest.log).

The completed CPU checks use an analytic mass-2 inelastic point impact: a body
starting 2 m above its floor at -3 m/s completes one second, reaches the floor,
and transfers exactly 6 Ns. Two replays match. Rejected copies deliberately
corrupt position, clock, controller generation and warm values to detect
leakage. Separate controls cover a later fatal branch, depth and attempt
exhaustion, backend exceptions, four invalid timesteps, and four odd FP32
durations whose rounded nonzero halves would change the total elapsed time.
The split requires `double(half)*2 == double(parent)`; underflow cannot silently
change the duration claimed by the ledger.

The driver keeps physical admission budgets unchanged. Only the caller's
explicit retryable failure bits permit subdivision. Any additional failure
bit rejects, even if an earlier diagnostic reports a retryable contact error.
State copies must own their storage; callbacks and controller samplers must
not mutate external physical state. This is a driver contract, not a guarantee
for aliased buffers or a stateful external callback.

The native whole-yarn finite-bench contact path now executes on Apple M4 Pro.
The focused Metal probe checks eleven geometry cases, two retained kinetic-energy
counterexamples, coupled normal/sliding/sticking contact, frictionless controls,
and exact state/contact replay. The fixed-step obscured-interior impact is
explicitly rejected instead of admitting its endpoint-clear but penetrated path.
All accepted unforced cases retain nonincreasing kinetic energy.

The opt-in adaptive driver now completes the retained obscured-interior impact
and its unequal-mass and reversed-order controls. Before an endpoint-only clip
can erase an earlier interior collision, the yarn owner rewinds both endpoints
to their common event, applies an inelastic velocity impulse there, and casts
the remaining common chord. It refuses to overwrite an earlier graph color's
position or velocity response. Unresolved coupled trajectories still reject.

The full 10 ms request still rejects as one fixed step. Transactional dyadic
subdivision completes it at depth 1, with two accepted leaves, exact state/grid
replay, nonincreasing kinetic energy, and zero penetration in the independent
FP64 sampled two-piece path audit. The native regression covers 11 geometry
cases and includes position/velocity ownership counterexamples. Contact ABI 17
retains the recovered event endpoints for this audit; earlier binaries and
buffers fail the existing ABI guard.

The prior depth-8 and depth-16 failures remain preserved below. Retry policy
stays depth 8 / 512 attempts, with whole-request rollback on unresolved failure.
These focused successful impacts do not establish arbitrary coupled graph
completion, full bag spill, throughput, reciprocal elastic fruit contact, or
complete work/reaction closure. The same final artifacts pass on M4 Pro and M6.

[Common-event source/binary receipt](assets/static-yarn-common-event-20261013/receipt.json),
[M4 build and regression execution](assets/static-yarn-common-event-20261013/adaptive-common-event-final-validation.log),
and [M6 execution](assets/static-yarn-common-event-20261013/adaptive-common-event-m6-validation.log).

```sh
cmake --build build --target numi-solver-transactional-substep
ctest --test-dir build -R '^static_contact.transactional_substep_host$' --output-on-failure
```

For the executed whole-yarn gate:

```sh
cmake --build build --target numi-solver-cloth-metal
ctest --test-dir build -R '^cloth_bag.static_yarn_native$' --output-on-failure
```

[Native source/binary receipt](assets/static-yarn-native-m4pro-20261013/receipt.json) and [executed gate](assets/static-yarn-native-m4pro-20261013/solver-final-native-gate.log). The historical [depth-8 failure](assets/static-yarn-native-m4pro-20261013/solver-adaptive-static-yarn-validation.log) and [depth-16 diagnostic failure](assets/static-yarn-native-m4pro-20261013/solver-refined-adaptive-validation.log) remain failed evidence for the former prediction path; they were not deleted or relabelled as passing.

The same executable and shader library also passed this focused gate on Apple
M6 (macOS 27.0, arm64). Their hashes match the M4 Pro build. This verifies native
Metal execution with the transferred artifacts; it does not qualify building
on M6, adaptive contact, or a performance improvement. The M6 host lacks the
Metal compiler. See the [M6 receipt](assets/static-yarn-native-m6-20261013/solver-m6-receipt.json),
[executed gate](assets/static-yarn-native-m6-20261013/solver-m6-native-gate.log), and
[host identity](assets/static-yarn-native-m6-20261013/solver-m6-host-identity.log).

The adaptive common-event regression is also available as `ctest --test-dir build -R ^cloth_bag.static_yarn_adaptive_native$ --output-on-failure`.
