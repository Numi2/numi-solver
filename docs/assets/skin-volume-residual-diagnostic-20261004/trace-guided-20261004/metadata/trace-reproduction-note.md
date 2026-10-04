# Trace-only replay note

The preregistered trace-guided candidate completed the first seven control steps, but the initial writer emitted FGMRES CSV only on failure. The numerical run is retained unchanged in output/candidate/. This same-setting replay changes no source solver policy, material, input, load, timestep, or gate; the instrumentation now also flushes on success. Its only purpose is to record the convergence history needed before another policy decision. The trace-reproduction output is separate and must not be counted as an independent replicate.
