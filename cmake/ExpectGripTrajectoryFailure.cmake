if(NOT DEFINED EXECUTABLE OR NOT DEFINED TRAJECTORY OR NOT DEFINED EXPECT)
    message(FATAL_ERROR "EXECUTABLE, TRAJECTORY, and EXPECT are required")
endif()

execute_process(
    COMMAND "${EXECUTABLE}"
        --scenario recorded
        --grip-trajectory "${TRAJECTORY}"
        --steps 1
        --substeps 1
        --iterations 1
        --replays 2
    RESULT_VARIABLE result
    OUTPUT_VARIABLE output
    ERROR_VARIABLE error
)

set(combined "${output}${error}")
if(NOT result EQUAL 2)
    message(FATAL_ERROR
        "invalid grip trajectory unexpectedly returned ${result}\n${combined}"
    )
endif()
if(NOT combined MATCHES "${EXPECT}")
    message(FATAL_ERROR
        "invalid grip trajectory did not report '${EXPECT}'\n${combined}"
    )
endif()

message(STATUS "invalid grip trajectory rejected: ${EXPECT}")
