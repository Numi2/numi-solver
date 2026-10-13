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
The focused Metal probe checks nine geometry cases, two retained kinetic-energy
counterexamples, coupled normal/sliding/sticking contact, frictionless controls,
and exact state/contact replay. The fixed-step obscured-interior impact is
explicitly rejected instead of admitting its endpoint-clear but penetrated path.
All accepted unforced cases retain nonincreasing kinetic energy.

The optional adaptive driver remains an **unqualified experiment**. On the
obscured-interior case its depth-8 run rejects 38 attempts, reaches 32 provisional
leaves, and rolls the complete requested interval back with zero committed time.
A depth-16 diagnostic also fails; increasing the retry budget is not a fix.
The production retry policy remains depth 8 and 512 attempts. The retained
failure and the last attempted interval are reported by the probe. This does
not establish successful adaptive native impact, complete bag spill, throughput,
reciprocal elastic fruit contact, or complete work/reaction closure.

```sh
cmake --build build --target numi-solver-transactional-substep
ctest --test-dir build -R '^static_contact.transactional_substep_host$' --output-on-failure
```

For the executed whole-yarn gate:

```sh
cmake --build build --target numi-solver-cloth-metal
ctest --test-dir build -R '^cloth_bag.static_yarn_native$' --output-on-failure
```

[Native source/binary receipt](assets/static-yarn-native-m4pro-20261013/receipt.json) and [executed gate](assets/static-yarn-native-m4pro-20261013/solver-final-native-gate.log). The retained [depth-8 failure](assets/static-yarn-native-m4pro-20261013/solver-adaptive-static-yarn-validation.log) and [depth-16 diagnostic failure](assets/static-yarn-native-m4pro-20261013/solver-refined-adaptive-validation.log) remain qualification failures.

The same executable and shader library also passed this focused gate on Apple
M6 (macOS 27.0, arm64). Their hashes match the M4 Pro build. This verifies native
Metal execution with the transferred artifacts; it does not qualify building
on M6, adaptive contact, or a performance improvement. The M6 host lacks the
Metal compiler. See the [M6 receipt](assets/static-yarn-native-m6-20261013/solver-m6-receipt.json),
[executed gate](assets/static-yarn-native-m6-20261013/solver-m6-native-gate.log), and
[host identity](assets/static-yarn-native-m6-20261013/solver-m6-host-identity.log).
