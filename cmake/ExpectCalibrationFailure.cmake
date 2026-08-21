if(NOT DEFINED EXECUTABLE OR
   NOT DEFINED PARAMETERS OR
   NOT DEFINED OBSERVATIONS OR
   NOT DEFINED EXPECT)
    message(FATAL_ERROR "calibration rejection checker is missing arguments")
endif()
if(NOT DEFINED EXPECTED_RESULT)
    set(EXPECTED_RESULT 1)
endif()

execute_process(
    COMMAND
        "${EXECUTABLE}"
        --parameters "${PARAMETERS}"
        --observations "${OBSERVATIONS}"
    RESULT_VARIABLE result
    OUTPUT_VARIABLE output
    ERROR_VARIABLE error
)

message("${output}${error}")

if(NOT result EQUAL EXPECTED_RESULT)
    message(FATAL_ERROR
        "expected calibration rejection exit ${EXPECTED_RESULT}, got ${result}"
    )
endif()

if(NOT "${output}${error}" MATCHES "${EXPECT}")
    message(FATAL_ERROR "calibration rejection did not report ${EXPECT}")
endif()
