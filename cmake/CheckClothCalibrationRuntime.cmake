if(NOT DEFINED CALIBRATOR OR NOT DEFINED CPU_EXECUTABLE OR
   NOT DEFINED METAL_EXECUTABLE OR NOT DEFINED PARAMETERS OR
   NOT DEFINED WORK_DIRECTORY)
    message(FATAL_ERROR
        "CALIBRATOR, CPU_EXECUTABLE, METAL_EXECUTABLE, PARAMETERS, and "
        "WORK_DIRECTORY are required"
    )
endif()

file(STRINGS "${PARAMETERS}" parameter_lines)
list(POP_FRONT parameter_lines)
set(parameter_names)
foreach(line IN LISTS parameter_lines)
    string(REPLACE "," ";" fields "${line}")
    list(GET fields 0 name)
    list(APPEND parameter_names "${name}")
endforeach()
list(LENGTH parameter_names parameter_count)
if(NOT parameter_count EQUAL 19)
    message(FATAL_ERROR "runtime contract must contain 19 parameters")
endif()

set(observation_text
    "trial_id,split,observable,measured,sigma,baseline"
)
foreach(name IN LISTS parameter_names)
    string(APPEND observation_text ",dlog_${name}")
endforeach()
string(APPEND observation_text "\n")

math(EXPR last_parameter "${parameter_count} - 1")
foreach(row RANGE 0 ${last_parameter})
    list(GET parameter_names ${row} name)
    set(sensitivities)
    foreach(column RANGE 0 ${last_parameter})
        if(row EQUAL column)
            string(APPEND sensitivities ",1.0")
        else()
            string(APPEND sensitivities ",0.0")
        endif()
    endforeach()
    string(APPEND observation_text
        "cal_${name},calibration,unit_${name},0.0,1.0,0.0"
        "${sensitivities}\n"
        "heldout_${name},heldout,unit_${name},0.0,1.0,0.0"
        "${sensitivities}\n"
    )
endforeach()

set(observations "${WORK_DIRECTORY}/full-contract-observations.csv")
set(material "${WORK_DIRECTORY}/full-contract-material.txt")
file(WRITE "${observations}" "${observation_text}")

execute_process(
    COMMAND "${CALIBRATOR}"
        --parameters "${PARAMETERS}"
        --observations "${observations}"
        --material-output "${material}"
        --allow-no-heldout-improvement
    RESULT_VARIABLE calibration_result
    OUTPUT_VARIABLE calibration_output
    ERROR_VARIABLE calibration_error
)
if(NOT calibration_result EQUAL 0 OR
   NOT calibration_output MATCHES "qualified=true")
    message(FATAL_ERROR
        "full-contract calibration failed (${calibration_result})\n"
        "${calibration_output}${calibration_error}"
    )
endif()

execute_process(
    COMMAND "${CPU_EXECUTABLE}"
        --steps 1
        --substeps 1
        --iterations 1
        --replays 2
        --material "${material}"
    RESULT_VARIABLE cpu_result
    OUTPUT_VARIABLE cpu_output
    ERROR_VARIABLE cpu_error
)
if(NOT cpu_result EQUAL 0 OR
   NOT cpu_output MATCHES
       "material_schema=numi.cloth.material.v1 material_artifact_loaded=true" OR
   NOT cpu_output MATCHES "result=PASS")
    message(FATAL_ERROR
        "CPU rejected generated material artifact (${cpu_result})\n"
        "${cpu_output}${cpu_error}"
    )
endif()

execute_process(
    COMMAND "${METAL_EXECUTABLE}"
        --replays 2
        --iterations 32
        --strain-sweeps 3
        --material "${material}"
    RESULT_VARIABLE metal_result
    OUTPUT_VARIABLE metal_output
    ERROR_VARIABLE metal_error
)
if(NOT metal_result EQUAL 0 OR
   NOT metal_output MATCHES
       "material_schema=numi.cloth.material.v1 material_artifact_loaded=true" OR
   NOT metal_output MATCHES "result=PASS")
    message(FATAL_ERROR
        "Metal rejected generated material artifact (${metal_result})\n"
        "${metal_output}${metal_error}"
    )
endif()

string(REGEX MATCH "parameters_hash=0x[0-9a-f]+" calibration_hash
    "${calibration_output}")
if(calibration_hash STREQUAL "" OR
   NOT cpu_output MATCHES "${calibration_hash}" OR
   NOT metal_output MATCHES "${calibration_hash}")
    message(FATAL_ERROR "calibration provenance did not reach both runtimes")
endif()

message(STATUS
    "full cloth calibration artifact reached CPU and Metal: "
    "${calibration_hash}"
)
