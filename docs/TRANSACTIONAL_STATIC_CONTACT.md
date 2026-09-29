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

The native finite-bench candidate compiles with an opt-in adaptive host path,
an authoritative grip-trajectory sampler, a retained attempt grid, exact replay
hashing of that grid and full-frame rollback. Its focused obscured-interior
yarn-impact test has **not executed on Metal**. Native continuous contact,
successful full bag replay, throughput, reciprocal elastic fruit contact and
complete work/reaction closure remain open. The active and queued GPU jobs
use their existing immutable binaries.

```sh
cmake --build build --target numi-solver-transactional-substep
ctest --test-dir build -R '^static_contact.transactional_substep_host$' --output-on-failure
```
