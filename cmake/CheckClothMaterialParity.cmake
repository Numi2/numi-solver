if(NOT DEFINED EXECUTABLE OR NOT DEFINED MATERIAL OR NOT DEFINED KIND)
    message(FATAL_ERROR "EXECUTABLE, MATERIAL, and KIND are required")
endif()

if(KIND STREQUAL "cpu")
    set(arguments
        --steps 1
        --substeps 1
        --iterations 1
        --replays 2
    )
elseif(KIND STREQUAL "metal")
    set(arguments
        --replays 2
        --iterations 32
        --strain-sweeps 3
    )
else()
    message(FATAL_ERROR "unknown cloth material parity kind: ${KIND}")
endif()

execute_process(
    COMMAND "${EXECUTABLE}" ${arguments}
    RESULT_VARIABLE default_result
    OUTPUT_VARIABLE default_output
    ERROR_VARIABLE default_error
)
if(NOT default_result EQUAL 0)
    message(FATAL_ERROR
        "authored-default run failed (${default_result})\n"
        "${default_output}${default_error}"
    )
endif()

execute_process(
    COMMAND "${EXECUTABLE}" ${arguments} --material "${MATERIAL}"
    RESULT_VARIABLE artifact_result
    OUTPUT_VARIABLE artifact_output
    ERROR_VARIABLE artifact_error
)
if(NOT artifact_result EQUAL 0)
    message(FATAL_ERROR
        "explicit-default run failed (${artifact_result})\n"
        "${artifact_output}${artifact_error}"
    )
endif()

string(REGEX MATCHALL "state_hash=0x[0-9a-f]+" default_hashes
    "${default_output}")
string(REGEX MATCHALL "state_hash=0x[0-9a-f]+" artifact_hashes
    "${artifact_output}")
if(default_hashes STREQUAL "" OR artifact_hashes STREQUAL "")
    message(FATAL_ERROR "cloth parity runs did not publish state hashes")
endif()
if(NOT default_hashes STREQUAL artifact_hashes)
    message(FATAL_ERROR
        "explicit defaults changed physical state\n"
        "authored=${default_hashes}\nartifact=${artifact_hashes}"
    )
endif()
if(NOT artifact_output MATCHES
   "material_schema=numi.cloth.material.v1 material_artifact_loaded=true")
    message(FATAL_ERROR "explicit-default run did not report loaded material")
endif()
if(NOT artifact_output MATCHES "parameters_hash=synthetic-defaults")
    message(FATAL_ERROR "explicit-default provenance was not reported")
endif()

message(STATUS "${KIND} cloth material default parity: ${default_hashes}")
