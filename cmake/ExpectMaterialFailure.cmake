if(NOT DEFINED EXECUTABLE OR NOT DEFINED MATERIAL OR NOT DEFINED EXPECT)
    message(FATAL_ERROR "EXECUTABLE, MATERIAL, and EXPECT are required")
endif()

execute_process(
    COMMAND "${EXECUTABLE}"
        --steps 1
        --substeps 1
        --iterations 1
        --replays 2
        --material "${MATERIAL}"
    RESULT_VARIABLE result
    OUTPUT_VARIABLE output
    ERROR_VARIABLE error
)

set(combined "${output}${error}")
if(NOT result EQUAL 1 AND NOT result EQUAL 2)
    message(FATAL_ERROR
        "invalid material unexpectedly returned ${result}\n${combined}"
    )
endif()
if(NOT combined MATCHES "${EXPECT}")
    message(FATAL_ERROR
        "invalid material did not report '${EXPECT}'\n${combined}"
    )
endif()

message(STATUS "invalid cloth material rejected: ${EXPECT}")
