if(NOT DEFINED CPU_EXECUTABLE OR NOT DEFINED METAL_EXECUTABLE OR
   NOT DEFINED IDENTITY_TRAJECTORY OR NOT DEFINED ROTATION_TRAJECTORY OR
   NOT DEFINED REGRAB_TRAJECTORY OR NOT DEFINED PATCH_REGRAB_TRAJECTORY)
    message(FATAL_ERROR
        "CPU_EXECUTABLE, METAL_EXECUTABLE, IDENTITY_TRAJECTORY, "
        "ROTATION_TRAJECTORY, REGRAB_TRAJECTORY, and "
        "PATCH_REGRAB_TRAJECTORY are required"
    )
endif()

set(cpu_arguments
    --scenario recorded
    --steps 1
    --substeps 24
    --iterations 32
    --replays 2
)
execute_process(
    COMMAND "${CPU_EXECUTABLE}" ${cpu_arguments}
        --grip-trajectory "${PATCH_REGRAB_TRAJECTORY}"
    RESULT_VARIABLE patch_regrab_result
    OUTPUT_VARIABLE patch_regrab_output
    ERROR_VARIABLE patch_regrab_error
)
if(NOT patch_regrab_result EQUAL 0 OR
   NOT patch_regrab_output MATCHES
       "grip_trajectory_schema=numi.grip.trajectory.v3" OR
   NOT patch_regrab_output MATCHES
       "attachment_generations=3 selection_mode=nearest_cuff_patch" OR
   NOT patch_regrab_output MATCHES "regrab_count=2" OR
   NOT patch_regrab_output MATCHES "patch_selection_count=2" OR
   NOT patch_regrab_output MATCHES "maximum_patch_ring_shift=12" OR
   NOT patch_regrab_output MATCHES "selected_patch_center_ring=24" OR
   NOT patch_regrab_output MATCHES
       "patch_particles_unique=true patch_topology_exact=true" OR
   NOT patch_regrab_output MATCHES "deterministic=true" OR
   NOT patch_regrab_output MATCHES "result=PASS")
    message(FATAL_ERROR
        "CPU spatial seam re-grab failed (${patch_regrab_result})\n"
        "${patch_regrab_output}${patch_regrab_error}"
    )
endif()
execute_process(
    COMMAND "${CPU_EXECUTABLE}" ${cpu_arguments}
        --grip-trajectory "${REGRAB_TRAJECTORY}"
    RESULT_VARIABLE regrab_result
    OUTPUT_VARIABLE regrab_output
    ERROR_VARIABLE regrab_error
)
if(NOT regrab_result EQUAL 0 OR
   NOT regrab_output MATCHES
       "grip_trajectory_schema=numi.grip.trajectory.v2" OR
   NOT regrab_output MATCHES "attachment_generations=3" OR
   NOT regrab_output MATCHES "regrab_count=2" OR
   NOT regrab_output MATCHES "inactive_grip_substeps=[1-9][0-9]*" OR
   NOT regrab_output MATCHES "deterministic=true" OR
   NOT regrab_output MATCHES "result=PASS")
    message(FATAL_ERROR
        "CPU seam re-grab trajectory failed (${regrab_result})\n"
        "${regrab_output}${regrab_error}"
    )
endif()
execute_process(
    COMMAND "${CPU_EXECUTABLE}" ${cpu_arguments}
        --grip-trajectory "${IDENTITY_TRAJECTORY}"
    RESULT_VARIABLE identity_result
    OUTPUT_VARIABLE identity_output
    ERROR_VARIABLE identity_error
)
if(NOT identity_result EQUAL 0 OR
   NOT identity_output MATCHES "deterministic=true" OR
   NOT identity_output MATCHES "result=PASS")
    message(FATAL_ERROR
        "CPU identity grip trajectory failed (${identity_result})\n"
        "${identity_output}${identity_error}"
    )
endif()

execute_process(
    COMMAND "${CPU_EXECUTABLE}" ${cpu_arguments}
        --grip-trajectory "${ROTATION_TRAJECTORY}"
    RESULT_VARIABLE rotation_result
    OUTPUT_VARIABLE rotation_output
    ERROR_VARIABLE rotation_error
)
if(NOT rotation_result EQUAL 0 OR
   NOT rotation_output MATCHES "deterministic=true" OR
   NOT rotation_output MATCHES "maximum_rotation_radians=0.0349" OR
   NOT rotation_output MATCHES "result=PASS")
    message(FATAL_ERROR
        "CPU rotating grip trajectory failed (${rotation_result})\n"
        "${rotation_output}${rotation_error}"
    )
endif()

string(REGEX MATCH "state_hash=0x[0-9a-f]+" identity_hash
    "${identity_output}")
string(REGEX MATCH "state_hash=0x[0-9a-f]+" rotation_hash
    "${rotation_output}")
if(identity_hash STREQUAL "" OR rotation_hash STREQUAL "" OR
   identity_hash STREQUAL rotation_hash)
    message(FATAL_ERROR
        "rotating the seam did not change the CPU physical state\n"
        "identity=${identity_hash} rotation=${rotation_hash}"
    )
endif()

execute_process(
    COMMAND "${METAL_EXECUTABLE}"
        --replays 2
        --iterations 32
        --strain-sweeps 3
        --grip-trajectory "${PATCH_REGRAB_TRAJECTORY}"
        --recorded-steps 1
        --recorded-dump-every 1
    RESULT_VARIABLE metal_result
    OUTPUT_VARIABLE metal_output
    ERROR_VARIABLE metal_error
)
if(NOT metal_result EQUAL 0 OR
   NOT metal_output MATCHES
       "grip_rotation_angle_radians=.*replay_exact=true failure_flags=0" OR
   NOT metal_output MATCHES
       "recorded_requested=true steps=1 replay_exact=true" OR
   NOT metal_output MATCHES
       "regrab_count=2 inactive_grip_substeps=[1-9][0-9]*" OR
   NOT metal_output MATCHES
       "attachment_generations_exact=true" OR
   NOT metal_output MATCHES
       "patch_selection_count=2 selected_patch_center_ring=24" OR
   NOT metal_output MATCHES
       "patch_center_exact=true patch_topology_exact=true" OR
   NOT metal_output MATCHES "qualified=true state_hash=0x[0-9a-f]+" OR
   NOT metal_output MATCHES "result=PASS")
    message(FATAL_ERROR
        "Metal rotating grip trajectory failed (${metal_result})\n"
        "${metal_output}${metal_error}"
    )
endif()

string(REGEX MATCH "content_fingerprint=0x[0-9a-f]+" cpu_fingerprint
    "${patch_regrab_output}")
if(cpu_fingerprint STREQUAL "" OR
   NOT metal_output MATCHES "${cpu_fingerprint}")
    message(FATAL_ERROR "grip trajectory provenance did not reach both paths")
endif()

message(STATUS
    "6-DoF seam trajectory changed CPU state, re-grabbed continuously, and "
    "passed Metal replay: "
    "${cpu_fingerprint}"
)
